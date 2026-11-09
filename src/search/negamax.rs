// Alpha-beta search: root, negamax, quiescence and draw detection.
use super::*;

/// Past this half-width the aspiration window opens fully.
pub(super) const ASP_FULL_WINDOW: i32 = 500;

/// `hashes` holds every position before this node; index `root` is the root position and
/// nothing before `floor` (the last null move) counts. One earlier occurrence after the
/// root is a draw; positions up to the root need two, as in a real three-fold.
#[inline]
pub(super) fn is_repetition(hash: u64, hmc: u16, hashes: &[u64], root: usize, floor: usize) -> bool {
    let n = hashes.len();
    let max_back = (hmc as usize).min(n - floor.min(n));
    let mut seen = false;
    let mut i = 2usize;
    while i <= max_back {
        let idx = n - i;
        if hashes[idx] == hash {
            if idx > root || seen {
                return true;
            }
            seen = true;
        }
        i += 2;
    }
    false
}

/// Only dead draws (K vs K, K+minor vs K); KN vs KN and similar are left to eval.
#[inline]
pub(super) fn is_insufficient_material(board: &Board) -> bool {
    let w = Color::White.idx();
    let b = Color::Black.idx();
    if board.piece_bb[w][PieceKind::Pawn.idx()]
        | board.piece_bb[b][PieceKind::Pawn.idx()]
        | board.piece_bb[w][PieceKind::Rook.idx()]
        | board.piece_bb[b][PieceKind::Rook.idx()]
        | board.piece_bb[w][PieceKind::Queen.idx()]
        | board.piece_bb[b][PieceKind::Queen.idx()]
        != 0
    {
        return false;
    }
    let minors = board.piece_bb[w][PieceKind::Knight.idx()]
        | board.piece_bb[w][PieceKind::Bishop.idx()]
        | board.piece_bb[b][PieceKind::Knight.idx()]
        | board.piece_bb[b][PieceKind::Bishop.idx()];
    minors.count_ones() <= 1
}

/// On fail high the new move is adopted at once; on fail low the previous best is kept.
pub(super) fn search_root_aspiration(
    board: &mut Board,
    depth: u32,
    prev_best: Option<Move>,
    prev_score: i32,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
) -> (Option<Move>, i32) {
    let p = ctx.params;
    if depth < p.asp_min_depth as u32 || prev_score.abs() >= MATE_THRESHOLD {
        return search_root(board, depth, prev_best, -INF, INF, ctx, tt);
    }

    let mut delta = p.asp_delta;
    let mut alpha = (prev_score - delta).max(-INF);
    let mut beta = (prev_score + delta).min(INF);
    let mut best = prev_best;
    loop {
        let (m, s) = search_root(board, depth, best, alpha, beta, ctx, tt);
        if ctx.stop {
            // A fail-low interrupted by time is not an upgrade, so keep the prior best.
            if s > alpha {
                if let Some(mm) = m {
                    best = Some(mm);
                }
            }
            return (best, s);
        }
        if s <= alpha {
            beta = (alpha + beta) / 2;
            alpha = (s - delta).max(-INF);
        } else if s >= beta {
            if let Some(mm) = m {
                best = Some(mm);
            }
            beta = (s + delta).min(INF);
        } else {
            if let Some(mm) = m {
                best = Some(mm);
            }
            return (best, s);
        }
        delta += delta / 2 + 1;
        if delta >= ASP_FULL_WINDOW {
            alpha = -INF;
            beta = INF;
        }
    }
}

#[inline]
pub(super) fn root_node_idx(m: Move) -> usize {
    m.from().idx() * 64 + m.to().idx()
}

pub(super) fn search_root(
    board: &mut Board,
    depth: u32,
    pv: Option<Move>,
    mut alpha: i32,
    beta: i32,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
) -> (Option<Move>, i32) {
    let mut moves = MoveList::new();
    board.gen_legal_into(&mut moves);
    if moves.is_empty() {
        let s = if board.in_check(board.side_to_move) { -MATE_SCORE } else { 0 };
        return (None, s);
    }
    let root_hash = board.hash;
    let original_alpha = alpha;
    ctx.pv_len[0] = 0;
    let raw_eval = if board.in_check(board.side_to_move) { None } else { Some(ctx.accs.evaluate(board)) };
    ctx.evals[0] = raw_eval.map(|ev| ctx.corrected_eval(board, ev));
    let tt_move = tt.probe(root_hash).map(|e| Move(e.move_packed));
    ctx.threats[0] = threat_map(board);
    let mut sm = ScoredMoves::from_list(&moves, board, pv, tt_move, ctx, 0);
    let mut best: Option<Move> = None;
    let mut best_score = -INF;
    let root_in_check = raw_eval.is_none();
    for i in 0..sm.len {
        let (m, _) = sm.pop_best(i);
        let nodes_before = ctx.nodes;
        let move_key = piece_key(board, m);
        let undo = board.make_move(m);
        ctx.played[1] = Some(move_key);
        ctx.accs.push_after(board, m, &undo);
        ctx.hashes.push(root_hash);
        let mut score;
        if i == 0 {
            score = -negamax(board, depth.saturating_sub(1), -beta, -alpha, 1, ctx, tt, true, None, false);
        } else {
            let full = depth.saturating_sub(1);
            let quiet = !m.is_capture() && !m.is_promotion();
            let mut reduced = full;
            if depth >= 3 && i >= 3 && quiet && !root_in_check && !board.in_check(board.side_to_move) {
                let r = ctx.lmr[(depth as usize).min(63)][(i + 1).min(63)] - ctx.params.lmr_root_offset;
                reduced = (full as i32 - r / 1024).clamp(1, full as i32) as u32;
            }
            score = -negamax(board, reduced, -alpha - 1, -alpha, 1, ctx, tt, true, None, true);
            if reduced < full && !ctx.stop && score > alpha {
                score = -negamax(board, full, -alpha - 1, -alpha, 1, ctx, tt, true, None, true);
            }
            if !ctx.stop && score > alpha && score < beta {
                score = -negamax(board, full, -beta, -alpha, 1, ctx, tt, true, None, false);
            }
        }
        ctx.hashes.pop();
        ctx.accs.pop();
        board.unmake_move(m, undo);
        ctx.root_nodes[root_node_idx(m)] += ctx.nodes - nodes_before;
        if ctx.stop {
            return (best, best_score);
        }
        if score > best_score {
            best_score = score;
            best = Some(m);
            ctx.update_pv(0, m);
            if score > alpha {
                alpha = score;
            }
            if score >= beta {
                break;
            }
        }
    }
    let bound = if best_score >= beta {
        BOUND_LOWER
    } else if best_score > original_alpha {
        BOUND_EXACT
    } else {
        BOUND_UPPER
    };
    if best.is_some() {
        tt.store(root_hash, best, tt_score_to_store(best_score, 0), raw_eval, depth.min(255) as u8, bound, true);
    }
    (best, best_score)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn negamax(
    board: &mut Board,
    mut depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
    allow_null: bool,
    excluded: Option<Move>,
    // Expected to fail high (a non-PV node whose parent expects to fail low).
    cut_node: bool,
) -> i32 {
    ctx.pv_len[ply as usize] = ply as usize;
    if ply as usize >= MAX_PLY {
        return ctx.accs.evaluate(board);
    }

    let original_alpha = alpha;
    let pv_node = beta - alpha > 1;
    let ply_i = ply as i32;
    let ply_us = ply as usize;
    let hash = board.hash;

    // Before the TT probe so a drawn line never returns a stale score; skipped at the root.
    if ply > 0 {
        if is_repetition(hash, board.halfmove_clock, &ctx.hashes, ctx.root_idx, ctx.rep_floor)
            || is_insufficient_material(board)
        {
            return 0;
        }
        if board.halfmove_clock >= 100 {
            // Checkmate takes precedence over the fifty-move rule.
            if !board.in_check(board.side_to_move) {
                return 0;
            }
            let mut ml = MoveList::new();
            board.gen_legal_into(&mut ml);
            return if ml.is_empty() { -MATE_SCORE + ply_i } else { 0 };
        }
    }

    // qsearch counts the node and checks the clock itself.
    if depth == 0 {
        return qsearch(board, alpha, beta, ply, ctx, tt);
    }
    if ctx.check() {
        return 0;
    }
    ctx.seldepth = ctx.seldepth.max(ply as usize);

    let mut tt_move: Option<Move> = None;
    let mut tt_score: Option<i32> = None;
    let mut tt_depth: u32 = 0;
    let mut tt_bound: u8 = BOUND_NONE;
    let mut tt_eval: Option<i32> = None;
    // On the PV now or at some earlier visit.
    let mut tt_pv = pv_node;
    // In a singular search the TT entry covers the full move set, so don't cut or overwrite it.
    if excluded.is_none() {
        if let Some(entry) = tt.probe(hash) {
            tt_pv |= entry.pv();
            if entry.move_packed != 0 {
                tt_move = Some(Move(entry.move_packed));
            }
            let s = tt_score_from_probe(entry.score, ply_i);
            tt_score = Some(s);
            tt_depth = entry.depth as u32;
            tt_bound = entry.bound();
            tt_eval = entry.eval();
            if !pv_node && tt_depth >= depth {
                match tt_bound {
                    BOUND_EXACT => return s,
                    BOUND_LOWER if s >= beta => return s,
                    BOUND_UPPER if s <= alpha => return s,
                    _ => {}
                }
            }
        }
    }

    // Without a TT move this node was never searched well; spend less on it.
    if depth >= IIR_MIN_DEPTH && tt_move.is_none() && excluded.is_none() {
        depth -= 1;
    }

    let in_check = board.in_check(board.side_to_move);
    let p = ctx.params;
    let raw_eval = if in_check { None } else { Some(tt_eval.unwrap_or_else(|| ctx.accs.evaluate(board))) };
    let static_eval = raw_eval.map(|ev| ctx.corrected_eval(board, ev));
    let eval = static_eval.map(|ev| adjust_eval(ev, tt_score, tt_bound));
    ctx.evals[ply_us] = static_eval;
    // Compare with our previous turn; fall back a full move if that node was in check.
    let improving = match static_eval {
        None => false,
        Some(ev) => {
            let back = |n: usize| if ply_us >= n { ctx.evals[ply_us - n] } else { None };
            back(2).or_else(|| back(4)).is_none_or(|e| ev > e)
        }
    };

    // Razoring: far below alpha at low depth, trust qsearch if it agrees.
    if !pv_node
        && !in_check
        && excluded.is_none()
        && depth <= p.razor_max_depth as u32
        && alpha.abs() < MATE_THRESHOLD
        && eval.unwrap() + p.razor_margin * depth as i32 <= alpha
    {
        let score = qsearch(board, alpha, alpha + 1, ply, ctx, tt);
        if ctx.stop {
            return alpha;
        }
        if score <= alpha {
            return score;
        }
    }

    if !pv_node
        && !in_check
        && depth >= 1
        && depth <= p.rfp_max_depth as u32
        && beta.abs() < MATE_THRESHOLD
        && eval.unwrap() - p.rfp_margin * (depth as i32 - improving as i32) >= beta
    {
        return beta;
    }

    if !pv_node && allow_null && !in_check && depth >= p.nmp_min_depth as u32 {
        let stm_idx = board.side_to_move.idx();
        let non_pawn_bb = board.piece_bb[stm_idx][PieceKind::Knight.idx()]
            | board.piece_bb[stm_idx][PieceKind::Bishop.idx()]
            | board.piece_bb[stm_idx][PieceKind::Rook.idx()]
            | board.piece_bb[stm_idx][PieceKind::Queen.idx()];
        if non_pawn_bb != 0 && eval.unwrap() >= beta {
            let r = p.nmp_base_r as u32 + (depth / p.nmp_divisor as u32);
            let undo = board.make_null_move();
            tt.prefetch(board.hash);
            ctx.accs.push_null();
            if ply_us + 1 < MAX_PLY {
                ctx.played[ply_us + 1] = None;
            }
            ctx.hashes.push(hash);
            let floor = std::mem::replace(&mut ctx.rep_floor, ctx.hashes.len());
            let null_score =
                -negamax(board, depth.saturating_sub(1 + r), -beta, -beta + 1, ply + 1, ctx, tt, false, None, false);
            ctx.rep_floor = floor;
            ctx.hashes.pop();
            ctx.accs.pop_null();
            board.unmake_null_move(undo);
            if ctx.stop {
                return alpha;
            }
            if null_score >= beta {
                return beta;
            }
        }
    }

    // ProbCut: a capture that clears beta by a margin at reduced depth very likely refutes this node.
    let pc_beta = beta + p.probcut_margin;
    if !pv_node
        && !in_check
        && excluded.is_none()
        && depth >= p.probcut_min_depth as u32
        && beta.abs() < MATE_THRESHOLD
        && !(tt_depth + 3 >= depth && tt_score.is_some_and(|s| s < pc_beta))
    {
        let se = static_eval.unwrap();
        let mut noisy = MoveList::new();
        board.gen_noisy_into(&mut noisy);
        let mut pc = ScoredMoves::empty();
        for k in 0..noisy.len() {
            let m = noisy.get(k);
            // Only moves whose material swing alone can bridge the gap to pc_beta.
            if see(board, m) >= pc_beta - se {
                pc.push(m, if Some(m) == tt_move { 1_000_000 } else { qscore(m, board) });
            }
        }
        let pc_depth = depth - 4;
        for k in 0..pc.len {
            let (m, _) = pc.pop_best(k);
            let key = piece_key(board, m);
            let undo = board.make_move(m);
            tt.prefetch(board.hash);
            if ply_us + 1 < MAX_PLY {
                ctx.played[ply_us + 1] = Some(key);
            }
            ctx.accs.push_after(board, m, &undo);
            ctx.hashes.push(hash);
            // A cheap qsearch filters out moves that don't even hold up tactically.
            let mut score = -qsearch(board, -pc_beta, -pc_beta + 1, ply + 1, ctx, tt);
            if score >= pc_beta && pc_depth > 0 && !ctx.stop {
                score = -negamax(board, pc_depth, -pc_beta, -pc_beta + 1, ply + 1, ctx, tt, true, None, false);
            }
            ctx.hashes.pop();
            ctx.accs.pop();
            board.unmake_move(m, undo);
            if ctx.stop {
                return alpha;
            }
            if score >= pc_beta {
                tt.store(hash, Some(m), tt_score_to_store(score, ply_i), raw_eval, (pc_depth + 1) as u8, BOUND_LOWER, tt_pv);
                return score;
            }
        }
    }

    let mut moves = MoveList::new();
    board.gen_legal_into(&mut moves);
    if moves.is_empty() {
        if in_check {
            return -MATE_SCORE + ply_i;
        }
        return 0;
    }

    ctx.threats[ply_us] = threat_map(board);
    let mut sm = ScoredMoves::tt_move_first(&moves, board, tt_move, ctx, ply_us);
    let mut best_move: Option<Move> = None;
    let mut best_score = -INF;
    let mut searched_quiets: [Move; 64] = [Move(0); 64];
    let mut sq_n: usize = 0;

    // Quiets dropped once pruning covers them all; they still count toward the move index.
    let mut skipped = 0usize;
    for slot in 0.. {
        if slot >= sm.len {
            break;
        }
        if slot == 1 && sm.deferred {
            sm.score_deferred(board, ctx, ply_us);
        }
        let i = slot + skipped;
        let (m, move_score) = sm.pop_best(slot);
        if Some(m) == excluded {
            continue;
        }

        // Never prune while best_score is a mated score, so forced lines survive.
        let is_quiet = !m.is_capture() && !m.is_promotion();
        let move_hist = if i > 0 && is_quiet { quiet_history(ctx, board, m, ply_us) } else { 0 };
        if i > 0 && !in_check && best_score > -MATE_THRESHOLD {
            if is_quiet
                && depth <= p.lmp_max_depth as u32
                && i as i32 >= (p.lmp_base + (depth as i32) * (depth as i32)) / (2 - improving as i32)
            {
                skipped += sm.drop_quiets_after(slot);
                continue;
            }
            // Only this move goes: a killer ahead of the history-sorted quiets may score lower than they do.
            if is_quiet
                && depth <= p.fp_max_depth as u32
                && eval.is_some_and(|ev| ev + p.fp_margin * depth as i32 + move_hist / p.fp_hist_div <= alpha)
            {
                continue;
            }
            if depth <= p.see_max_depth as u32 {
                let margin = if is_quiet { p.see_quiet_margin } else { p.see_capt_margin };
                // Past the TT move a capture's score already holds its SEE.
                let see_val = if m.is_capture() { see_from_score(move_score) } else { see(board, m) };
                if see_val < -(margin * depth as i32) {
                    continue;
                }
            }
        }
        let move_key = piece_key(board, m);

        // Extend a lone TT move, multi-cut if another move beats beta, otherwise reduce it.
        let mut extension: i32 = 0;
        if excluded.is_none()
            && Some(m) == tt_move
            && depth >= p.singular_min_depth as u32
            && tt_depth + 3 >= depth
            && (tt_bound == BOUND_LOWER || tt_bound == BOUND_EXACT)
        {
            if let Some(ts) = tt_score {
                if ts.abs() < MATE_THRESHOLD {
                    let singular_beta = ts - p.singular_margin * depth as i32;
                    let reduced = (depth - 1) / 2;
                    let value = negamax(
                        board,
                        reduced,
                        singular_beta - 1,
                        singular_beta,
                        ply,
                        ctx,
                        tt,
                        false,
                        Some(m),
                        cut_node,
                    );
                    // The excluded search shares this ply's PV slot.
                    ctx.pv_len[ply_us] = ply_us;
                    if ctx.stop {
                        return best_score;
                    }
                    if value < singular_beta {
                        extension = 1;
                    } else if singular_beta >= beta {
                        return singular_beta;
                    } else if ts >= beta {
                        extension = -2;
                    } else if !pv_node {
                        extension = -1;
                    }
                }
            }
        }
        let new_depth = ((depth as i32 - 1) + extension).max(0) as u32;

        let undo = board.make_move(m);
        tt.prefetch(board.hash);
        if ply_us + 1 < MAX_PLY {
            ctx.played[ply_us + 1] = Some(move_key);
        }
        ctx.accs.push_after(board, m, &undo);
        ctx.hashes.push(hash);
        if is_quiet && sq_n < searched_quiets.len() {
            searched_quiets[sq_n] = m;
            sq_n += 1;
        }

        let mut score;
        if i == 0 {
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1, ctx, tt, true, None, !pv_node && !cut_node);
        } else {
            // Captures are reduced too, except in PV nodes.
            let noisy = m.is_capture() || m.is_promotion();
            let do_lmr = depth >= p.lmr_min_depth as u32
                && i >= p.lmr_min_move as usize + pv_node as usize
                && !(pv_node && noisy);
            let mut reduced = new_depth;
            if do_lmr {
                let mut r = ctx.lmr[(depth as usize).min(63)][(i + 1).min(63)];
                if noisy {
                    r -= p.lmr_capt_offset;
                } else {
                    r -= (move_hist * 1024 / p.lmr_history_div).clamp(-3072, 3072);
                    if ctx.killers[ply_us].contains(&Some(m)) {
                        r -= p.lmr_killer;
                    }
                }
                r += p.lmr_not_improving * !improving as i32;
                r += p.lmr_cut_node * cut_node as i32;
                r -= p.lmr_tt_pv * tt_pv as i32;
                r -= p.lmr_check * board.in_check(board.side_to_move) as i32;
                r += p.lmr_tt_capture * tt_move.is_some_and(|t| t.is_capture()) as i32;
                reduced = (new_depth as i32 - r / 1024).clamp(1, new_depth.max(1) as i32) as u32;
            }

            score = -negamax(board, reduced, -alpha - 1, -alpha, ply + 1, ctx, tt, true, None, do_lmr || !cut_node);

            // A reduced move that beat alpha is re-searched, deeper if it looks clearly
            // better than the best so far and shallower if barely.
            let mut full = new_depth;
            if reduced < new_depth && !ctx.stop && score > alpha {
                if best_score > -INF {
                    if score > best_score + p.lmr_deeper_margin {
                        full += 1;
                    } else if score < best_score + p.lmr_shallower_margin {
                        full -= 1;
                    }
                }
                if full > reduced {
                    score = -negamax(board, full, -alpha - 1, -alpha, ply + 1, ctx, tt, true, None, !cut_node);
                }
            }

            if !ctx.stop && score > alpha && score < beta {
                score = -negamax(board, full, -beta, -alpha, ply + 1, ctx, tt, true, None, false);
            }
        }

        ctx.hashes.pop();
        ctx.accs.pop();

        board.unmake_move(m, undo);
        if ctx.stop {
            return best_score;
        }
        if score > best_score {
            best_score = score;
            best_move = Some(m);
        }
        if score >= beta {
            if !m.is_capture() && !m.is_promotion() {
                ctx.update_killers(ply_us, m);
                ctx.update_quiet_histories(board, ply_us, m, &searched_quiets[..sq_n], depth);
            }
            if excluded.is_none() {
                tt.store(
                    hash,
                    Some(m),
                    tt_score_to_store(best_score, ply_i),
                    raw_eval,
                    depth.min(255) as u8,
                    BOUND_LOWER,
                    tt_pv,
                );
                if let (Some(raw), Some(se)) = (raw_eval, static_eval) {
                    ctx.update_correction(board, raw, se, best_score, Some(m), BOUND_LOWER, depth);
                }
            }
            return best_score;
        }
        if score > alpha {
            alpha = score;
            ctx.update_pv(ply_us, m);
        }
    }

    let bound = if best_score > original_alpha { BOUND_EXACT } else { BOUND_UPPER };
    if excluded.is_none() {
        tt.store(
            hash,
            if bound == BOUND_UPPER { None } else { best_move },
            tt_score_to_store(best_score, ply_i),
            raw_eval,
            depth.min(255) as u8,
            bound,
            tt_pv,
        );
        if let (Some(raw), Some(se)) = (raw_eval, static_eval) {
            ctx.update_correction(board, raw, se, best_score, best_move, bound, depth);
        }
    }
    best_score
}

pub(super) fn qsearch(board: &mut Board, mut alpha: i32, beta: i32, ply: u32, ctx: &mut SearchCtx, tt: &mut Tt) -> i32 {
    if ctx.check() {
        return 0;
    }
    ctx.seldepth = ctx.seldepth.max(ply as usize);

    let pv_node = beta - alpha > 1;
    let original_alpha = alpha;
    let hash = board.hash;
    let ply_i = ply as i32;

    let mut tt_move: Option<Move> = None;
    let mut tt_score: Option<i32> = None;
    let mut tt_bound = BOUND_NONE;
    let mut tt_eval: Option<i32> = None;
    let mut tt_pv = pv_node;
    if let Some(entry) = tt.probe(hash) {
        tt_pv |= entry.pv();
        if entry.move_packed != 0 {
            tt_move = Some(Move(entry.move_packed));
        }
        let s = tt_score_from_probe(entry.score, ply_i);
        tt_score = Some(s);
        tt_bound = entry.bound();
        tt_eval = entry.eval();
        if !pv_node {
            match tt_bound {
                BOUND_EXACT => return s,
                BOUND_LOWER if s >= beta => return s,
                BOUND_UPPER if s <= alpha => return s,
                _ => {}
            }
        }
    }

    let in_check = board.in_check(board.side_to_move);

    let mut best_score = -INF;
    let mut raw_eval: Option<i32> = None;
    if !in_check {
        let raw = tt_eval.unwrap_or_else(|| ctx.accs.evaluate(board));
        raw_eval = Some(raw);
        let stand_pat = adjust_eval(ctx.corrected_eval(board, raw), tt_score, tt_bound);
        best_score = stand_pat;
        if stand_pat >= beta {
            if tt_bound == BOUND_NONE {
                tt.store(hash, None, tt_score_to_store(stand_pat, ply_i), raw_eval, 0, BOUND_LOWER, tt_pv);
            }
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    if (ply as usize) >= MAX_PLY {
        return ctx.accs.evaluate(board);
    }

    // Out of check only captures and promotions are searched, so only those are generated.
    let mut moves = MoveList::new();
    if in_check {
        board.gen_legal_into(&mut moves);
        if moves.is_empty() {
            return -MATE_SCORE + ply as i32;
        }
    } else {
        board.gen_noisy_into(&mut moves);
        if moves.is_empty() && !board.has_legal_move() {
            return 0;
        }
    }

    let mut q = ScoredMoves::empty();
    for i in 0..moves.len() {
        let m = moves.get(i);
        if !in_check && m.is_capture() && !capture_cannot_lose(board, m) && see(board, m) < 0 {
            continue;
        }
        let s = if Some(m) == tt_move { 1_000_000 } else { qscore(m, board) };
        q.push(m, s);
    }

    let mut best_move: Option<Move> = None;
    for i in 0..q.len {
        let (m, _) = q.pop_best(i);

        let undo = board.make_move(m);
        tt.prefetch(board.hash);
        ctx.accs.push_after(board, m, &undo);
        let score = -qsearch(board, -beta, -alpha, ply + 1, ctx, tt);
        ctx.accs.pop();
        board.unmake_move(m, undo);
        if ctx.stop {
            return best_score;
        }
        if score > best_score {
            best_score = score;
            best_move = Some(m);
        }
        if score >= beta {
            tt.store(hash, Some(m), tt_score_to_store(best_score, ply_i), raw_eval, 0, BOUND_LOWER, tt_pv);
            return best_score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    let bound = if best_score > original_alpha { BOUND_EXACT } else { BOUND_UPPER };
    tt.store(hash, best_move, tt_score_to_store(best_score, ply_i), raw_eval, 0, bound, tt_pv);
    best_score
}
