//! Точная арифметика для Джарвиса: безопасный вычислитель выражений.
//!
//! Маленькая модель (0.8B–2B) ненадёжна в арифметике: «17% от 23400»,
//! «128*46» она считает с ошибками. Здесь всё детерминировано: рекурсивный
//! спуск без внешних зависимостей и без `eval` — только числа, скобки и
//! арифметика. Фразы («сколько будет 128*46», «15% от 2400») вытаскивает
//! [`math_answer`], движок вызывает его до роутера — ответ мгновенный и
//! всегда верный.

/// Ошибка разбора или вычисления.
#[derive(Debug)]
pub struct CalcError(pub String);

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Вычислить арифметическое выражение: `+ - * / % ^`, скобки, унарный минус,
/// десятичные точки. Деление на ноль — ошибка.
pub fn eval(expr: &str) -> Result<f64, CalcError> {
    let cleaned: String = expr
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| match c {
            ',' => '.',
            '×' | 'x' | 'X' => '*',
            '÷' | ':' => '/',
            '−' | '–' | '—' => '-',
            other => other,
        })
        .collect();
    let mut p = Parser {
        chars: cleaned.chars().collect(),
        pos: 0,
    };
    let value = p.expr()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(CalcError(format!(
            "лишние символы после выражения: {}",
            p.chars[p.pos..].iter().collect::<String>()
        )));
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CalcError("результат не число".into()))
    }
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: char) -> bool {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// expr := term (('+' | '-') term)*
    fn expr(&mut self) -> Result<f64, CalcError> {
        let mut value = self.term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('+') => {
                    self.pos += 1;
                    value += self.term()?;
                }
                Some('-') => {
                    self.pos += 1;
                    value -= self.term()?;
                }
                _ => return Ok(value),
            }
        }
    }

    /// term := factor (('*' | '/' | '%') factor)*
    fn term(&mut self) -> Result<f64, CalcError> {
        let mut value = self.factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    // '**' — степень, как в Python.
                    if self.peek() == Some('*') {
                        self.pos += 1;
                        value = value.powf(self.factor()?);
                    } else {
                        value *= self.factor()?;
                    }
                }
                Some('/') => {
                    self.pos += 1;
                    let divisor = self.factor()?;
                    if divisor == 0.0 {
                        return Err(CalcError("деление на ноль".into()));
                    }
                    value /= divisor;
                }
                // '%' между числами — остаток от деления; «X% от Y» разбирает
                // math_answer до вызова сюда, сюда он не доходит.
                Some('%') => {
                    self.pos += 1;
                    let divisor = self.factor()?;
                    if divisor == 0.0 {
                        return Err(CalcError("деление на ноль".into()));
                    }
                    value %= divisor;
                }
                _ => return Ok(value),
            }
        }
    }

    /// factor := ('-' | '+') factor | power
    fn factor(&mut self) -> Result<f64, CalcError> {
        self.skip_ws();
        match self.peek() {
            Some('-') => {
                self.pos += 1;
                Ok(-self.factor()?)
            }
            Some('+') => {
                self.pos += 1;
                self.factor()
            }
            _ => self.power(),
        }
    }

    /// power := atom ('^' factor)? — правоассоциативная степень.
    fn power(&mut self) -> Result<f64, CalcError> {
        let base = self.atom()?;
        self.skip_ws();
        if self.peek() == Some('^') {
            self.pos += 1;
            Ok(base.powf(self.factor()?))
        } else {
            Ok(base)
        }
    }

    /// atom := число | '(' expr ')'
    fn atom(&mut self) -> Result<f64, CalcError> {
        self.skip_ws();
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let value = self.expr()?;
                if !self.eat(')') {
                    return Err(CalcError("не закрыта скобка".into()));
                }
                Ok(value)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => self.number(),
            Some(c) => Err(CalcError(format!("неожиданный символ «{c}»"))),
            None => Err(CalcError("выражение оборвалось".into())),
        }
    }

    fn number(&mut self) -> Result<f64, CalcError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.') {
            self.pos += 1;
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<f64>()
            .map_err(|_| CalcError(format!("не число: «{text}»")))
    }
}

/// Результат быстрой математики: фраза для озвучки.
pub struct MathAnswer {
    pub speech: String,
}

/// «Сколько будет 128*46», «посчитай (120+45)*3», «15% от 2400»,
/// «17 процентов от 23400» → мгновенный точный ответ. `None` — не математика
/// (или в фразе нет ни одной цифры: «посчитай, сколько лет я проживу» —
/// это разговор).
pub fn math_answer(prompt: &str) -> Option<MathAnswer> {
    let t = prompt.trim();
    if t.chars().filter(|c| c.is_ascii_digit()).count() == 0 {
        return None;
    }
    if t.chars().count() > 80 {
        return None;
    }

    // «P% от V» / «P процентов от V» / «P% of V».
    let lower = t.to_lowercase().replace('ё', "е");
    for marker in ["% от ", " процентов от ", "% of ", " процент от "] {
        if let Some(pos) = lower.find(marker) {
            let left = &lower[..pos];
            let right = &lower[pos + marker.len()..];
            let percent = trailing_number(left)?;
            let base = leading_number(right)?;
            let value = base * percent / 100.0;
            return Some(MathAnswer {
                speech: format!(
                    "{} процентов от {} — {}",
                    fmt(percent),
                    fmt(base),
                    fmt(value)
                ),
            });
        }
    }

    // «сколько будет EXPR», «посчитай EXPR», «вычисли EXPR», «EXPR = ?».
    // Ворота против ложных захватов: «проверь 10-15 человек» — это не
    // «10 минус 15». Нужен маркер намерения либо почти чистое выражение
    // без русских слов.
    let marker = [
        "сколько будет",
        "посчитай",
        "вычисли",
        "чему равно",
        "сколько составит",
        "сложи",
        "умножь",
        "подели",
        "раздели",
    ]
    .iter()
    .any(|m| lower.contains(m));
    let letters = t.chars().filter(|c| c.is_alphabetic() && !c.is_ascii()).count();
    let pure_expr = letters == 0;
    if !marker && !pure_expr {
        return None;
    }
    let expr = extract_expr(t)?;
    let value = eval(&expr).ok()?;
    Some(MathAnswer {
        speech: format!("{} — {}", expr.trim(), fmt(value)),
    })
}

/// Число, заканчивающееся на этой позиции (для левой части «% от»).
fn trailing_number(text: &str) -> Option<f64> {
    let digits: String = text
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '.' || *c == ' ')
        .collect::<String>()
        .chars()
        .rev()
        .filter(|c| !c.is_whitespace())
        .map(|c| if c == ',' { '.' } else { c })
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Число, с которого начинается строка (для правой части «% от»).
fn leading_number(text: &str) -> Option<f64> {
    let digits: String = text
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '.')
        .map(|c| if c == ',' { '.' } else { c })
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Вытащить выражение из фразы: чистая арифметика с цифрами и операторами,
/// "без букв вокруг (кроме служебных слов «сколько будет», «посчитай»…).
fn extract_expr(prompt: &str) -> Option<String> {
    let cleaned: String = prompt
        .chars()
        .filter(|c| c.is_ascii_digit() || "+-*/%^().,".contains(*c) || c.is_whitespace() || *c == '=')
        .collect();
    let expr: String = cleaned
        .replace('=', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("");
    // Должны остаться хотя бы две цифры и один оператор — иначе это не
    // арифметика («посчитай до десяти» цифр не содержит, отсеялось раньше).
    let has_digit = expr.chars().any(|c| c.is_ascii_digit());
    let has_op = expr.chars().any(|c| "+-*/%^".contains(c));
    (has_digit && has_op && expr.len() >= 3).then_some(expr)
}

/// Формат числа для озвучки: целое без точки, дробное с двумя знаками.
fn fmt(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e15 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Публичная озвучка результата для инструмента calc.
pub fn speech_number(v: f64) -> String {
    fmt(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_s(expr: &str) -> f64 {
        eval(expr).unwrap()
    }

    #[test]
    fn basic_arithmetic() {
        assert_eq!(eval_s("2+2"), 4.0);
        assert_eq!(eval_s("128*46"), 5888.0);
        assert_eq!(eval_s("(120+45)*3"), 495.0);
        assert_eq!(eval_s("10/4"), 2.5);
        assert_eq!(eval_s("2^10"), 1024.0);
        assert_eq!(eval_s("-5+3"), -2.0);
        assert_eq!(eval_s("10%3"), 1.0);
        assert_eq!(eval_s("3,5 + 1,5"), 5.0);
    }

    #[test]
    fn precedence_and_parens() {
        assert_eq!(eval_s("2+2*2"), 6.0);
        assert_eq!(eval_s("(2+2)*2"), 8.0);
        assert_eq!(eval_s("2*(3+4)"), 14.0);
        assert_eq!(eval_s("-(3+4)"), -7.0);
        assert_eq!(eval_s("2^3^2"), 512.0);
    }

    #[test]
    fn errors() {
        assert!(eval("1/0").is_err());
        assert!(eval("(1+2").is_err());
        assert!(eval("1 + пирог").is_err());
        assert!(eval("").is_err());
    }

    #[test]
    fn percent_phrases() {
        let a = math_answer("сколько будет 15% от 2400").unwrap();
        assert!(a.speech.contains("360"), "{}", a.speech);
        let a = math_answer("17 процентов от 23400").unwrap();
        assert!(a.speech.contains("3978"), "{}", a.speech);
    }

    #[test]
    fn expression_phrases() {
        let a = math_answer("сколько будет 128*46").unwrap();
        assert!(a.speech.contains("5888"), "{}", a.speech);
        let a = math_answer("посчитай (120+45)*3").unwrap();
        assert!(a.speech.contains("495"), "{}", a.speech);
    }

    #[test]
    fn non_math_passes_through() {
        assert!(math_answer("посчитай, сколько лет я проживу").is_none());
        assert!(math_answer("напиши мне стихотворение").is_none());
        assert!(math_answer("привет").is_none());
        // Слова-числа без цифр — разговор, не быстрый путь.
        assert!(math_answer("сколько будет два плюс два").is_none());
        // Диапазон с цифрами без маркера намерения — не «минус».
        assert!(math_answer("проверь 10-15 человек").is_none());
    }

    #[test]
    fn bare_expression_and_markers() {
        // Чистое выражение без слов —_math без маркера.
        assert!(math_answer("128*46").is_some());
        let a = math_answer("чему равно 2^10").unwrap();
        assert!(a.speech.contains("1024"), "{}", a.speech);
    }
}
