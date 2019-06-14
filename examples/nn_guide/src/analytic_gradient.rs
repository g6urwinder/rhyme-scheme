extern crate rand;
use rand::Rng;
use base;

pub fn analytic_gradient() {
    
    fn forwardMultiplyGate(x: f64, y: f64) -> f64 {
        x*y
    };
    
    fn forwardAddGate(x: f64, y: f64) -> f64 {
        x + y
    };

    let mut x: f64 = -2.0;
    let mut y: f64 = 3.0;
    let out = forwardMultiplyGate(x, y);
    let x_gradient = y;
    let y_gradient = x;

    let step_size = 0.01;
    x = x + step_size*x_gradient;
    y = y + step_size*y_gradient;
    let out_new = forwardMultiplyGate(x, y);

    println!("OLD OUTPUT #{:?} NEW OUTPUT {:?}", out, out_new);

    fn forwardCircuit(x: f64, y: f64, z: f64) -> f64 {
        let q = forwardAddGate(x,y);
        let f = forwardMultiplyGate(q, z);
        f
    }

    let output_forward_new = forwardCircuit(-2.0, 5.0, -4.0);
    println!("NEW OUTPUT WITH FORWARD + * #{:?}", output_forward_new);

    x = -2.0;
    y = 5.0;
    let mut z = -4.0;

    let mut q = forwardAddGate(x, y);
    let mut f = forwardMultiplyGate(q, z);

    let mut derivative_f_wrt_z = q;
    let mut derivative_f_wrt_q = z;

    let mut derivative_q_wrt_x = 1.0;
    let mut derivative_q_wrt_y = 1.0;

    let derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q;
    let derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q;

    let gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z];

    let step_size = 0.01;
    x = x + step_size * derivative_f_wrt_x;
    y = y + step_size * derivative_f_wrt_y;
    z = z + step_size * derivative_f_wrt_z;

    let new_q = forwardAddGate(x, y);
    println!("NEW GATE {:?}", new_q);

    let new_f = forwardMultiplyGate(q, z);
    println!("NEW F {:?}", new_f);
    
    struct Unit {
        value: f64,
        grad: f64,
    }

    impl Unit {
        fn new(value: f64, grad: f64) -> Unit {

           Unit {
            value: value,
            grad: grad,
           } 
        }
    }

    struct MultiplyGate {
        u0: Unit,
        u1: Unit,
        utop: Unit,
    }

    impl MultiplyGate {
        
        fn forward(&mut self, u0: Unit, u1: Unit) -> &mut MultiplyGate {
            
            self.u0 = u0;
            self.u1 = u1;
            self.utop = Unit::new(u0.value * u1.value, 0.0);
            self
        }

        fn backward(&mut self) -> &mut MultiplyGate {
            
            self.u0.grad = self.u0.grad + self.u1.value*self.utop.grad;
            self.u1.grad = self.u1.grad + self.u0.value*self.utop.grad;
            self
        }
    }

    struct AddGate {
        u0: Unit,
        u1: Unit,
        utop: Unit,
    }

    impl AddGate {
        
        fn forward(&mut self, u0: Unit, u1: Unit) -> &mut AddGate {
            
            self.u0 = u0;
            self.u1 = u1;
            self.utop = Unit::new(u0.value + u1.value, 0.0);
            self
        }

        fn backward(&mut self) -> &mut AddGate {
        
            self.u0.grad = self.u0.grad + 1.0 * self.utop.grad;
            self.u1.grad = self.u1.grad + 1.0 * self.utop.grad;
            self
        }
    }

    struct SigmoidGate {
        sig: Fn(i32) -> i32,
    }

    impl SigmoidGate {
        
        
    }
}

#[cfg(test)]
mod tests {
    

    use super::*;
    
    #[test]
    fn test_analytic_gradient() {
        
        analytic_gradient()
    }
}