// Tunable search parameters and their SPSA ranges.

pub struct ParamSpec {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

/// Divisor minimums are 2 so the formulas dividing by them never hit zero.
pub const PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec { name: "RFPMargin", default: 98, min: 20, max: 200 },
    ParamSpec { name: "RFPMaxDepth", default: 6, min: 3, max: 12 },
    ParamSpec { name: "NMPBaseR", default: 3, min: 1, max: 5 },
    ParamSpec { name: "NMPDivisor", default: 3, min: 2, max: 8 },
    ParamSpec { name: "NMPMinDepth", default: 2, min: 2, max: 6 },
    // LMR table: (LMRBase + ln(depth) * ln(move number) / LMRDiv) plies, both in hundredths.
    ParamSpec { name: "LMRBase", default: 57, min: 0, max: 200 },
    ParamSpec { name: "LMRDiv", default: 248, min: 100, max: 400 },
    ParamSpec { name: "LMRMinDepth", default: 2, min: 2, max: 6 },
    // Move index from which LMR applies; one later in PV nodes.
    ParamSpec { name: "LMRMinMove", default: 1, min: 1, max: 8 },
    ParamSpec { name: "LMRHistoryDiv", default: 8279, min: 1024, max: 32768 },
    // LMR adjustments in 1/1024 ply.
    ParamSpec { name: "LMRCaptOffset", default: 1253, min: 0, max: 3072 },
    ParamSpec { name: "LMRNotImproving", default: 702, min: 0, max: 2048 },
    ParamSpec { name: "LMRCutNode", default: 2262, min: 0, max: 4096 },
    ParamSpec { name: "LMRTtPv", default: 912, min: 0, max: 3072 },
    ParamSpec { name: "LMRCheck", default: 1150, min: 0, max: 2048 },
    ParamSpec { name: "LMRKiller", default: 1025, min: 0, max: 2048 },
    ParamSpec { name: "LMRTtCapture", default: 1037, min: 0, max: 2048 },
    ParamSpec { name: "LMRRootOffset", default: 1847, min: 0, max: 4096 },
    // After a reduced fail-high, re-search a ply deeper/shallower by score vs best (cp).
    ParamSpec { name: "LMRDeeperMargin", default: 58, min: 0, max: 200 },
    ParamSpec { name: "LMRShallowerMargin", default: 9, min: 0, max: 60 },
    // A *MaxDepth of 0 disables that pruning.
    ParamSpec { name: "LMPMaxDepth", default: 2, min: 0, max: 12 },
    ParamSpec { name: "LMPBase", default: 9, min: 1, max: 16 },
    ParamSpec { name: "FPMaxDepth", default: 6, min: 0, max: 12 },
    ParamSpec { name: "FPMargin", default: 118, min: 30, max: 400 },
    // Quiet history counts toward the futility margin as history / FPHistDiv cp.
    ParamSpec { name: "FPHistDiv", default: 18, min: 8, max: 256 },
    ParamSpec { name: "RazorMaxDepth", default: 3, min: 0, max: 6 },
    ParamSpec { name: "RazorMargin", default: 220, min: 100, max: 600 },
    ParamSpec { name: "SEEMaxDepth", default: 3, min: 0, max: 12 },
    ParamSpec { name: "SEEQuietMargin", default: 106, min: 0, max: 200 },
    ParamSpec { name: "SEECaptMargin", default: 103, min: 0, max: 200 },
    ParamSpec { name: "ProbCutMinDepth", default: 5, min: 3, max: 10 },
    ParamSpec { name: "ProbCutMargin", default: 202, min: 80, max: 400 },
    ParamSpec { name: "SingularMinDepth", default: 6, min: 4, max: 12 },
    ParamSpec { name: "SingularMargin", default: 3, min: 1, max: 8 },
    ParamSpec { name: "AspMinDepth", default: 6, min: 2, max: 10 },
    ParamSpec { name: "AspDelta", default: 21, min: 4, max: 60 },
];

#[derive(Clone, Copy)]
pub struct SearchParams {
    pub rfp_margin: i32,
    pub rfp_max_depth: i32,
    pub nmp_base_r: i32,
    pub nmp_divisor: i32,
    pub nmp_min_depth: i32,
    pub lmr_base: i32,
    pub lmr_div: i32,
    pub lmr_min_depth: i32,
    pub lmr_min_move: i32,
    pub lmr_history_div: i32,
    pub lmr_capt_offset: i32,
    pub lmr_not_improving: i32,
    pub lmr_cut_node: i32,
    pub lmr_tt_pv: i32,
    pub lmr_check: i32,
    pub lmr_killer: i32,
    pub lmr_tt_capture: i32,
    pub lmr_root_offset: i32,
    pub lmr_deeper_margin: i32,
    pub lmr_shallower_margin: i32,
    pub lmp_max_depth: i32,
    pub lmp_base: i32,
    pub fp_max_depth: i32,
    pub fp_margin: i32,
    pub fp_hist_div: i32,
    pub razor_max_depth: i32,
    pub razor_margin: i32,
    pub see_max_depth: i32,
    pub see_quiet_margin: i32,
    pub see_capt_margin: i32,
    pub probcut_min_depth: i32,
    pub probcut_margin: i32,
    pub singular_min_depth: i32,
    pub singular_margin: i32,
    pub asp_min_depth: i32,
    pub asp_delta: i32,
}

impl Default for SearchParams {
    fn default() -> Self {
        // Must match the PARAM_SPECS defaults.
        Self {
            rfp_margin: 98,
            rfp_max_depth: 6,
            nmp_base_r: 3,
            nmp_divisor: 3,
            nmp_min_depth: 2,
            lmr_base: 57,
            lmr_div: 248,
            lmr_min_depth: 2,
            lmr_min_move: 1,
            lmr_history_div: 8279,
            lmr_capt_offset: 1253,
            lmr_not_improving: 702,
            lmr_cut_node: 2262,
            lmr_tt_pv: 912,
            lmr_check: 1150,
            lmr_killer: 1025,
            lmr_tt_capture: 1037,
            lmr_root_offset: 1847,
            lmr_deeper_margin: 58,
            lmr_shallower_margin: 9,
            lmp_max_depth: 2,
            lmp_base: 9,
            fp_max_depth: 6,
            fp_margin: 118,
            fp_hist_div: 18,
            razor_max_depth: 3,
            razor_margin: 220,
            see_max_depth: 3,
            see_quiet_margin: 106,
            see_capt_margin: 103,
            probcut_min_depth: 5,
            probcut_margin: 202,
            singular_min_depth: 6,
            singular_margin: 3,
            asp_min_depth: 6,
            asp_delta: 21,
        }
    }
}

impl SearchParams {
    pub fn set(&mut self, name: &str, value: i32) -> bool {
        let spec = match PARAM_SPECS.iter().find(|s| s.name == name) {
            Some(s) => s,
            None => return false,
        };
        let v = value.clamp(spec.min, spec.max);
        match name {
            "RFPMargin" => self.rfp_margin = v,
            "RFPMaxDepth" => self.rfp_max_depth = v,
            "NMPBaseR" => self.nmp_base_r = v,
            "NMPDivisor" => self.nmp_divisor = v,
            "NMPMinDepth" => self.nmp_min_depth = v,
            "LMRBase" => self.lmr_base = v,
            "LMRDiv" => self.lmr_div = v,
            "LMRCaptOffset" => self.lmr_capt_offset = v,
            "LMRNotImproving" => self.lmr_not_improving = v,
            "LMRCutNode" => self.lmr_cut_node = v,
            "LMRTtPv" => self.lmr_tt_pv = v,
            "LMRCheck" => self.lmr_check = v,
            "LMRKiller" => self.lmr_killer = v,
            "LMRTtCapture" => self.lmr_tt_capture = v,
            "LMRRootOffset" => self.lmr_root_offset = v,
            "LMRDeeperMargin" => self.lmr_deeper_margin = v,
            "LMRShallowerMargin" => self.lmr_shallower_margin = v,
            "LMRMinDepth" => self.lmr_min_depth = v,
            "LMRMinMove" => self.lmr_min_move = v,
            "LMRHistoryDiv" => self.lmr_history_div = v,
            "LMPMaxDepth" => self.lmp_max_depth = v,
            "LMPBase" => self.lmp_base = v,
            "FPMaxDepth" => self.fp_max_depth = v,
            "FPMargin" => self.fp_margin = v,
            "FPHistDiv" => self.fp_hist_div = v,
            "RazorMaxDepth" => self.razor_max_depth = v,
            "RazorMargin" => self.razor_margin = v,
            "SEEMaxDepth" => self.see_max_depth = v,
            "SEEQuietMargin" => self.see_quiet_margin = v,
            "SEECaptMargin" => self.see_capt_margin = v,
            "ProbCutMinDepth" => self.probcut_min_depth = v,
            "ProbCutMargin" => self.probcut_margin = v,
            "SingularMinDepth" => self.singular_min_depth = v,
            "SingularMargin" => self.singular_margin = v,
            "AspMinDepth" => self.asp_min_depth = v,
            "AspDelta" => self.asp_delta = v,
            _ => return false,
        }
        true
    }
}
