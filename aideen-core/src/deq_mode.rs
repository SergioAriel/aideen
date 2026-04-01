#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeqSolveMode {
    DeqOnly,
    NoMamba,
    InitMamba,
    HistGated,
    FixedMamba,
}

impl DeqSolveMode {
    fn env_flag(name: &str) -> Option<bool> {
        std::env::var(name).ok().map(|v| {
            let vl = v.trim().to_ascii_lowercase();
            vl == "1" || vl == "true" || vl == "yes"
        })
    }

    pub fn from_env() -> Self {
        if let Ok(mode) = std::env::var("AIDEEN_DEQ_MODE") {
            match mode.trim().to_ascii_lowercase().as_str() {
                "deq_only" | "clean" | "clean_deq" => return Self::DeqOnly,
                "no_mamba" | "history_off" | "hist_off" => return Self::NoMamba,
                "init_mamba" | "hist_init" => return Self::InitMamba,
                "fixed_mamba" | "fixed_hist" => return Self::FixedMamba,
                "hist_gated" | "history_on" | "hist_on" => return Self::HistGated,
                _ => {}
            }
        }

        if Self::env_flag("AIDEEN_DEQ_ONLY") == Some(true) {
            return Self::DeqOnly;
        }
        if Self::env_flag("AIDEEN_DEQ_NO_MAMBA") == Some(true) {
            return Self::NoMamba;
        }
        if Self::env_flag("AIDEEN_DEQ_INIT_MAMBA") == Some(true) {
            return Self::InitMamba;
        }
        if Self::env_flag("AIDEEN_DEQ_FIXED_MAMBA") == Some(true) {
            return Self::FixedMamba;
        }

        match Self::env_flag("AIDEEN_DEQ_HIST_GATED") {
            Some(false) => Self::NoMamba,
            Some(true) => Self::HistGated,
            None => Self::HistGated,
        }
    }

    pub fn residual_alpha(self) -> f32 {
        if let Some(alpha) = std::env::var("AIDEEN_DEQ_RESIDUAL_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
        {
            return alpha;
        }

        match self {
            Self::DeqOnly => -2.0,
            Self::NoMamba => -1.0,
            Self::InitMamba => -0.75,
            Self::HistGated => -0.5,
            Self::FixedMamba => -0.25,
        }
    }

    pub fn is_clean_core(self) -> bool {
        matches!(self, Self::DeqOnly)
    }

    pub fn history_is_gated(self) -> bool {
        matches!(self, Self::HistGated)
    }
}
