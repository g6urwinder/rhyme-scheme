extern crate rand;
use rand::Rng;

fn numerical_gradient() {
    
    fn forwardMultiplyGate(x: f64,y: f64) -> f64 { 
        x*y 
    };
    
    let x: f64 = -2.0;
    let y: f64 = 3.0;
    let out: f64 = forwardMultiplyGate(x, y);
    let h = 0.0001;

    let xph = x + h;
    let out2 = forwardMultiplyGate(xph, y);
    let x_derivative = (out2 - out)/h;

    let yph = y + h;
    let out3 = forwardMultiplyGate(x, yph);
    let y_derivative = (out3 - out)/h;

    let step_size = 0.02;
    let x_try = x + step_size * x_derivative;
    let y_try = y + step_size * y_derivative;
    let out_new = forwardMultiplyGate(x_try, y_try);

    println!(" X_TRY {:?} Y_TRY {:?} OUT_NEW {:?}", x_try, y_try, out_new)
}

#[cfg(test)]
mod tests {
    
    use super::*;
    
    #[test]
    fn test_numerical_gradient() {
        
        numerical_gradient()
    }
}
