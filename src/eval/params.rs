use once_cell::sync::Lazy;

pub const N_HIDDEN: usize = 512;
pub const N_BUCKETS: usize = 8;

// Fixed by the embedded net (checked against its header when decoding), so the hot
// paths divide and branch on constants.
// Activations clamp to [0, QA]; output cp = sum * OUT_SCALE / (QA * QB).
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const OUT_SCALE: i32 = 400;
// Squared clipped ReLU; SCReLU sums are divided by QA to get back to QA * QB.
pub const SCRELU: bool = true;
// Kings on files e-h see the board flipped left-right.
pub const MIRRORED: bool = true;
// Own king square (from that side's view) -> king bucket. For files a-d by rank:
// [0,0,1,1, 2 x 8, 3 x 20], mirrored onto e-h.
pub const KING_BUCKET: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        let half = (sq / 8) * 4 + if sq % 8 > 3 { 7 - sq % 8 } else { sq % 8 };
        t[sq] = if half < 2 { 0 } else if half < 4 { 1 } else if half < 12 { 2 } else { 3 };
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
    pub ft_weights: Box<[[i16; N_HIDDEN]]>,
    pub ft_bias: [i16; N_HIDDEN],
    pub head_weights: [[i8; 2 * N_HIDDEN]; N_BUCKETS],
    // i16 copy for the SCReLU head, which multiplies in 16-bit lanes.
    pub head_weights_i16: [[i16; 2 * N_HIDDEN]; N_BUCKETS],
    pub head_bias_q: [i32; N_BUCKETS],
}

static RAW: &[u8] = include_bytes!("net.nnue");

pub(crate) static WEIGHTS: Lazy<NnueWeights> = Lazy::new(decode_weights);

fn decode_weights() -> NnueWeights {
    let read_i32 = |i: usize| {
        i32::from_le_bytes([RAW[i], RAW[i + 1], RAW[i + 2], RAW[i + 3]])
    };
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
    assert!(activation == 0 || activation == 1, "net.nnue: unknown activation {activation}");

    // Layout covers files a-d of each rank; the e-h half reuses it mirrored.
    const HEADER: usize = 32 + 32;
    let mut king_bucket = [0u8; 64];
    for (sq, b) in king_bucket.iter_mut().enumerate() {
        let file = sq % 8;
        *b = RAW[32 + (sq / 8) * 4 + file.min(7 - file)];
        assert!((*b as usize) < input_buckets, "net.nnue: king bucket out of range");
    }

    let ft_rows = 768 * input_buckets;
    let expected = HEADER + 4 * N_BUCKETS + (ft_rows * N_HIDDEN + N_HIDDEN) * 2 + N_BUCKETS * 2 * N_HIDDEN;
    assert_eq!(RAW.len(), expected, "net.nnue: unexpected size");

    let mut head_bias_q = [0i32; N_BUCKETS];
    for (b, bias) in head_bias_q.iter_mut().enumerate() {
        *bias = read_i32(HEADER + 4 * b);
    }

    let read_i16 = |i: usize| i16::from_le_bytes([RAW[i], RAW[i + 1]]);
    let mut cursor = HEADER + 4 * N_BUCKETS;

    let mut ft = vec![[0i16; N_HIDDEN]; ft_rows].into_boxed_slice();
    for row in ft.iter_mut() {
        for w in row.iter_mut() {
            *w = read_i16(cursor);
            cursor += 2;
        }
    }

    let mut ft_bias = [0i16; N_HIDDEN];
    for h in 0..N_HIDDEN {
        ft_bias[h] = read_i16(cursor);
        cursor += 2;
    }

    let mut head_weights = [[0i8; 2 * N_HIDDEN]; N_BUCKETS];
    for bucket in head_weights.iter_mut() {
        for w in bucket.iter_mut() {
            *w = RAW[cursor] as i8;
            cursor += 1;
        }
    }

    let head_weights_i16 = head_weights.map(|bucket| bucket.map(i16::from));

    assert!(
        qa == QA && qb == QB && out_scale == OUT_SCALE && (activation == 1) == SCRELU
            && mirrored == MIRRORED && king_bucket == KING_BUCKET,
        "net.nnue header does not match the compiled constants in params.rs"
    );

    NnueWeights {
        ft_weights: ft, ft_bias, head_weights, head_weights_i16,
        head_bias_q,
    }
}
