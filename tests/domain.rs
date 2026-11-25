//! Проверки доменной логики: период суток, разбор батареи, решения о подтверждении.

use turnoff_gpui::battery;
use turnoff_gpui::power::PowerAction;
use turnoff_gpui::scene::TimeOfDay;

#[test]
fn период_суток_задаётся_локальным_часом() {
    assert_eq!(TimeOfDay::from_hour(5), TimeOfDay::Morning);
    assert_eq!(TimeOfDay::from_hour(10), TimeOfDay::Morning);
    assert_eq!(TimeOfDay::from_hour(11), TimeOfDay::Day);
    assert_eq!(TimeOfDay::from_hour(16), TimeOfDay::Day);
    assert_eq!(TimeOfDay::from_hour(17), TimeOfDay::Evening);
    assert_eq!(TimeOfDay::from_hour(20), TimeOfDay::Evening);
    assert_eq!(TimeOfDay::from_hour(21), TimeOfDay::Night);
    assert_eq!(TimeOfDay::from_hour(4), TimeOfDay::Night);
}

#[test]
fn периоды_суток_перебираются_по_кругу() {
    assert_eq!(TimeOfDay::Morning.next(), TimeOfDay::Day);
    assert_eq!(TimeOfDay::Day.next(), TimeOfDay::Evening);
    assert_eq!(TimeOfDay::Evening.next(), TimeOfDay::Night);
    assert_eq!(TimeOfDay::Night.next(), TimeOfDay::Morning);
}

#[test]
fn у_каждого_периода_свои_слои_сцены() {
    let paths = [
        TimeOfDay::Morning.sky(),
        TimeOfDay::Day.sky(),
        TimeOfDay::Evening.sky(),
        TimeOfDay::Night.sky(),
    ];
    let unique: std::collections::HashSet<_> = paths.iter().collect();
    assert_eq!(unique.len(), paths.len(), "слои неба должны различаться");
}

#[test]
fn заряд_батареи_читается_из_вывода_pmset() {
    let info = battery::parse(
        "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=12345)\t87%; discharging; 3:41 remaining present: true",
    )
    .expect("заряд должен читаться");

    assert_eq!(info.percent, 87);
    assert!(!info.charging);
}

#[test]
fn зарядка_от_сети_распознаётся_корректно() {
    let info = battery::parse(
        "Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345)\t42%; charging; 1:02 remaining present: true",
    )
    .expect("заряд должен читаться");

    assert_eq!(info.percent, 42);
    assert!(info.charging);
}

#[test]
fn без_батареи_заряд_не_читается() {
    assert!(battery::parse("Now drawing from 'AC Power'\n").is_none());
}

#[test]
fn необратимые_действия_требуют_подтверждения() {
    assert!(PowerAction::Shutdown.needs_confirmation());
    assert!(PowerAction::Restart.needs_confirmation());
    assert!(!PowerAction::Sleep.needs_confirmation());
    assert!(!PowerAction::Hibernate.needs_confirmation());
}

#[test]
fn у_каждого_действия_своё_имя_и_подсказка() {
    let labels: std::collections::HashSet<_> = PowerAction::ALL
        .iter()
        .map(|action| action.label())
        .collect();
    assert_eq!(labels.len(), PowerAction::ALL.len());

    for action in PowerAction::ALL {
        assert!(!action.hint().is_empty());
        assert!(!action.confirm_title().is_empty());
    }
}
