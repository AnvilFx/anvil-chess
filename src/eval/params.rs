use once_cell::sync::Lazy;

pub const N_HIDDEN: usize = 1024;
pub const N_BUCKETS: usize = 8;

// Fixed by the embedded net (checked against its header when decoding), so the hot
// paths divide and branch on constants.
// Activations clamp to [0, QA]; output cp = sum * OUT_SCALE / (QA * QB).
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const OUT_SCALE: i32 = 400;
// Squared clipped ReLU; SCReLU sums are divided by QA to get back to QA * QB.
pub const SCRELU: bool = true;
// Dense nets: pairwise CReLU accumulator -> L1 (i8 weights) -> L2 -> 1 per output bucket, f32 after L1.
pub const DENSE: bool = false;
pub const L1: usize = 16;
pub const L2: usize = 32;
// Kings on files e-h see the board flipped left-right.
pub const MIRRORED: bool = true;
// Own king square (from that side's view) -> king bucket. For files a-d by rank:
// [0,0,1,1, 2 x 8, 3 x 20], mirrored onto e-h.
pub const KING_BUCKET: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        let half = (sq / 8) * 4 + if sq % 8 > 3 { 7 - sq % 8 } else { sq % 8 };
        t[sq] = if half < 2 {
            0
        } else if half < 4 {
            1
        } else if half < 12 {
            2
        } else {
            3
        };
        sq += 1;
    }
    t
};

// Output bucket by piece count, matching bullet's MaterialCount<N_BUCKETS>.
#[inline]
pub fn output_bucket(occupancy: u64) -> usize {
    (occupancy.count_ones() as usize - 2) / 32usize.div_ceil(N_BUCKETS)
}

pub struct NnueWeights {
    // 768 rows per king bucket.
    pub ft_weights: crate::mem::HugeBuf<[i16; N_HIDDEN]>,
    pub ft_bias: [i16; N_HIDDEN],
    pub head_weights: [[i8; 2 * N_HIDDEN]; N_BUCKETS],
    // i16 copy for the SCReLU head, which multiplies in 16-bit lanes.
    pub head_weights_i16: [[i16; 2 * N_HIDDEN]; N_BUCKETS],
    pub head_bias_q: [i32; N_BUCKETS],
    // All zero for a plain net; one element.
    pub dense: crate::mem::HugeBuf<DenseWeights>,
}

#[derive(Clone, Copy)]
pub struct DenseWeights {
    // [bucket][input group of 4][output * 4 + input in group], so one group's weights for all
    // L1 outputs are contiguous and a sparse pass can skip all-zero groups.
    pub l1w: [[[i8; 4 * L1]; N_HIDDEN / 4]; N_BUCKETS],
    // Per row: back from (pairwise input at QA^2 / 512) x (row's i8 scale) to float.
    pub l1_dequant: [[f32; L1]; N_BUCKETS],
    pub l1b: [[f32; L1]; N_BUCKETS],
    // [bucket][input][output], transposed from the file's [output][input].
    pub l2w: [[[f32; L2]; L1]; N_BUCKETS],
    pub l2b: [[f32; L2]; N_BUCKETS],
    pub l3w: [[f32; L2]; N_BUCKETS],
    pub l3b: [f32; N_BUCKETS],
}

static RAW: &[u8] = include_bytes!("net.nnue");

pub(crate) static WEIGHTS: Lazy<NnueWeights> = Lazy::new(decode_weights);

fn decode_weights() -> NnueWeights {
    let read_i32 = |i: usize| i32::from_le_bytes([RAW[i], RAW[i + 1], RAW[i + 2], RAW[i + 3]]);
    let n_hidden = read_i32(0);
    let qa = read_i32(4);
    let qb = read_i32(8);
    let n_buckets = read_i32(12);
    let out_scale = read_i32(16);
    let activation = read_i32(20);
    let input_buckets = read_i32(24) as usize;
    let mirrored = read_i32(28) != 0;
    assert_eq!(
        n_hidden as usize, N_HIDDEN,
        "net.nnue header n_hidden ({}) != compiled N_HIDDEN ({}); rebuild with a matching net",
        n_hidden, N_HIDDEN
    );
    assert_eq!(
        n_buckets as usize, N_BUCKETS,
        "net.nnue header n_buckets ({}) != compiled N_BUCKETS ({}); regenerate with tools/gen_net.py",
        n_buckets, N_BUCKETS
    );
    assert!((0..=2).contains(&activation), "net.nnue: unknown activation {activation}");
    let dense = activation == 2;

    // Layout covers files a-d of each rank; the e-h half reuses it mirrored.
    const HEADER: usize = 32 + 32;
    let mut king_bucket = [0u8; 64];
    for (sq, b) in king_bucket.iter_mut().enumerate() {
        let file = sq % 8;
        *b = RAW[32 + (sq / 8) * 4 + file.min(7 - file)];
        assert!((*b as usize) < input_buckets, "net.nnue: king bucket out of range");
    }

    let ft_rows = 768 * input_buckets;
    let ft_bytes = (ft_rows * N_HIDDEN + N_HIDDEN) * 2;
    let expected = if dense {
        assert!(
            read_i32(HEADER) as usize == L1 && read_i32(HEADER + 4) as usize == L2,
            "net.nnue: dense layer sizes don't match the compiled L1 / L2"
        );
        HEADER + 8 + ft_bytes + N_BUCKETS * L1 * N_HIDDEN + 4 * N_BUCKETS * (2 * L1 + L2 * L1 + L2 + L2 + 1)
    } else {
        HEADER + 4 * N_BUCKETS + ft_bytes + N_BUCKETS * 2 * N_HIDDEN
    };
    assert_eq!(RAW.len(), expected, "net.nnue: unexpected size");

    let mut head_bias_q = [0i32; N_BUCKETS];
    if !dense {
        for (b, bias) in head_bias_q.iter_mut().enumerate() {
            *bias = read_i32(HEADER + 4 * b);
        }
    }

    let read_i16 = |i: usize| i16::from_le_bytes([RAW[i], RAW[i + 1]]);
    let mut cursor = if dense { HEADER + 8 } else { HEADER + 4 * N_BUCKETS };

    let mut ft = unsafe { crate::mem::HugeBuf::<[i16; N_HIDDEN]>::zeroed(ft_rows) };
    for row in ft.iter_mut() {
        for w in row.iter_mut() {
            *w = read_i16(cursor);
            cursor += 2;
        }
    }

    let mut ft_bias = [0i16; N_HIDDEN];
    for b in ft_bias.iter_mut() {
        *b = read_i16(cursor);
        cursor += 2;
    }

    let mut head_weights = [[0i8; 2 * N_HIDDEN]; N_BUCKETS];
    // All-zero bytes are valid i8 / f32.
    let mut dw = unsafe { crate::mem::HugeBuf::<DenseWeights>::zeroed(1) };
    if dense {
        let d = &mut dw[0];
        for bucket in 0..N_BUCKETS {
            for j in 0..L1 {
                for i in 0..N_HIDDEN {
                    d.l1w[bucket][i / 4][j * 4 + i % 4] = RAW[cursor] as i8;
                    cursor += 1;
                }
            }
        }
        let mut f32s = RAW[cursor..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let mut fill = |dst: &mut dyn Iterator<Item = &mut f32>| {
            for v in dst {
                *v = f32s.next().expect("net.nnue: dense block too short");
            }
        };
        fill(&mut d.l1_dequant.iter_mut().flatten());
        for s in d.l1_dequant.iter_mut().flatten() {
            *s = 512.0 / ((QA * QA) as f32 * *s);
        }
        fill(&mut d.l1b.iter_mut().flatten());
        let mut l2w = [[[0f32; L1]; L2]; N_BUCKETS];
        fill(&mut l2w.iter_mut().flatten().flatten());
        for (b, rows) in l2w.iter().enumerate() {
            for (k, row) in rows.iter().enumerate() {
                for (j, &w) in row.iter().enumerate() {
                    d.l2w[b][j][k] = w;
                }
            }
        }
        fill(&mut d.l2b.iter_mut().flatten());
        fill(&mut d.l3w.iter_mut().flatten());
        fill(&mut d.l3b.iter_mut());
    } else {
        for bucket in head_weights.iter_mut() {
            for w in bucket.iter_mut() {
                *w = RAW[cursor] as i8;
                cursor += 1;
            }
        }
    }

    let head_weights_i16 = head_weights.map(|bucket| bucket.map(i16::from));

    assert!(
        qa == QA
            && (dense || qb == QB)
            && out_scale == OUT_SCALE
            && dense == DENSE
            && (dense || (activation == 1) == SCRELU)
            && mirrored == MIRRORED
            && king_bucket == KING_BUCKET,
        "net.nnue header does not match the compiled constants in params.rs"
    );

    NnueWeights { ft_weights: ft, ft_bias, head_weights, head_weights_i16, head_bias_q, dense: dw }
}
