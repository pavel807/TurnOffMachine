//! Фоновая сцена, выбираемая по времени суток.

use crate::assets::AppIcon;

/// Слой сцены. Порядок объявления = порядок наложения.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SceneLayer {
    Sky,
    Terrain,
    Ground,
}

/// Период суток.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeOfDay {
    /// 5:00–10:59.
    Morning,
    /// 11:00–16:59.
    Day,
    /// 17:00–20:59.
    Evening,
    /// 21:00–4:59.
    Night,
}

impl TimeOfDay {
    pub const ALL: [Self; 4] = [Self::Morning, Self::Day, Self::Evening, Self::Night];

    /// Период по локальному часу.
    pub fn from_hour(hour: u32) -> Self {
        match hour {
            5..=10 => Self::Morning,
            11..=16 => Self::Day,
            17..=20 => Self::Evening,
            _ => Self::Night,
        }
    }

    /// Следующий период по кругу.
    pub fn next(self) -> Self {
        match self {
            Self::Morning => Self::Day,
            Self::Day => Self::Evening,
            Self::Evening => Self::Night,
            Self::Night => Self::Morning,
        }
    }

    pub fn icon(self) -> AppIcon {
        match self {
            Self::Morning => AppIcon::Sunrise,
            Self::Day => AppIcon::Sun,
            Self::Evening => AppIcon::Sunset,
            Self::Night => AppIcon::CloudMoon,
        }
    }

    pub fn sky(self) -> &'static str {
        match self {
            Self::Morning => "sky-morning.svg",
            Self::Day => "sky-day.svg",
            Self::Evening => "sky-evening.svg",
            Self::Night => "sky-night.svg",
        }
    }

    pub fn terrain(self) -> &'static str {
        match self {
            Self::Morning => "terrain-morning.svg",
            Self::Day => "terrain-day.svg",
            Self::Evening => "terrain-evening.svg",
            Self::Night => "terrain-night.svg",
        }
    }

    pub fn ground(self) -> &'static str {
        match self {
            Self::Morning => "ground-morning.svg",
            Self::Day => "ground-day.svg",
            Self::Evening => "ground-evening.svg",
            Self::Night => "ground-night.svg",
        }
    }

    /// Слои по порядку наложения.
    pub fn layers(self) -> [(SceneLayer, &'static str); 3] {
        [
            (SceneLayer::Sky, self.sky()),
            (SceneLayer::Terrain, self.terrain()),
            (SceneLayer::Ground, self.ground()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_hour_boundaries() {
        assert_eq!(TimeOfDay::from_hour(4), TimeOfDay::Night);
        assert_eq!(TimeOfDay::from_hour(5), TimeOfDay::Morning);
        assert_eq!(TimeOfDay::from_hour(11), TimeOfDay::Day);
        assert_eq!(TimeOfDay::from_hour(17), TimeOfDay::Evening);
        assert_eq!(TimeOfDay::from_hour(21), TimeOfDay::Night);
    }

    #[test]
    fn cycles_through_all_periods() {
        let mut period = TimeOfDay::Morning;
        for _ in 0..4 {
            period = period.next();
        }
        assert_eq!(period, TimeOfDay::Morning);
    }
}
