extern crate rand;
use rand::Rng;

fn single_gate_circuit() {
    
    fn forwardMultiplyGate(x: f64,y: f64) -> f64 { 
        x*y 
    };
    
    let mut rng = rand::thread_rng();
    let x: f64 = -2.0;
    let y: f64 = 3.0;
    let tweak_amount: f64 = 0.01;
    let mut best_out: f64 = std::i32::MIN as f64;
    let mut best_x: f64 = x;
    let mut best_y: f64 = y;

    for _ in 0..10 {
        
        let gen_1: f64 = rng.gen_range(0f64, 1f64);
        let gen_2: f64 = rng.gen_range(0f64, 1f64);
        let mut x_try: f64 = x as f64 + tweak_amount * ((gen_1*2.0 as f64) - 1.0 as f64);
        let mut y_try: f64 = y as f64 + tweak_amount * ((gen_2*2.0 as f64) - 1.0 as f64);
        let out: f64 = forwardMultiplyGate(x_try, y_try);

        if (out > best_out) {
            best_out = out;
            best_x = x_try;
            best_y = y_try;
        }

        println!("{:?}, #{:?}, #{:?}", best_out, best_x, best_y);
    }
    
    println!("BEST OUT #{:?}", best_out)
}

#[cfg(test)]
mod tests {
    
    use super::*;
    
    #[test]
    fn test_single_gate_circuit() {
        
        single_gate_circuit()
    }
}
