"""
Simple Linear Congruential Generator (LCG)
Recurrence: X_{n+1} = (a * X_n + b) mod m

Parameters:
 - seed : initial integer seed (X0)
 - a    : multiplier (integer)
 - b    : increment (integer)
 - m    : modulus (integer, default 2**32)

Provides:
 - next_int(): next X_n (integer in [0, m-1])
 - next_float(): next U_n in [0, 1)
 - generate_ints(n) / generate_floats(n)
 - reset(seed=None)
"""

from typing import List, Optional

class LinearCongruentialGenerator:
    def __init__(self, seed: int, a: int, b: int, m: int = 2**32):
        if m <= 0:
            raise ValueError("Modulus m must be a positive integer.")
        # store parameters (force ints)
        self.m = int(m)
        self.a = int(a)
        self.b = int(b)
        self.seed = int(seed) % self.m
        self.state = int(self.seed)
    
    def next_int(self) -> int:
        """Advance generator and return next integer X_{n+1} in [0, m-1]."""
        self.state = (self.a * self.state + self.b) % self.m
        return self.state
    
    def next_float(self) -> float:
        """Return next normalized float U in [0,1)."""
        return self.next_int() / self.m
    
    def generate_ints(self, n: int) -> List[int]:
        """Generate n integers (raw outputs)."""
        return [self.next_int() for _ in range(int(n))]
    
    def generate_floats(self, n: int) -> List[float]:
        """Generate n normalized floats in [0,1)."""
        return [self.next_float() for _ in range(int(n))]
    
    def reset(self, seed: Optional[int] = None):
        """Reset the internal state to given seed or original seed if None."""
        if seed is None:
            self.state = int(self.seed)
        else:
            self.seed = int(seed) % self.m
            self.state = int(self.seed)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example parameters (common choice for 32-bit LCG: Numerical Recipes style)
    seed = 10
    a = 16
    b = 10
    m = 42

    lcg = LinearCongruentialGenerator(seed=seed, a=a, b=b, m=m)

    # print first 10 integers
    ints = lcg.generate_ints(20)
    print("First 10 integers (X_n):", ints)

    # reset and print first 10 normalized floats
    lcg.reset(seed)
    floats = lcg.generate_floats(20)
    print("First 10 floats (U_n in [0,1)):", floats)
