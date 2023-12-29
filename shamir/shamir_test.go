package shamir

import (
	"math"
	"math/big"
	"testing"
)

func Test_encrypt(t *testing.T) {
	k := 2
	n := 3
	p := math.Pow(2, 61) - 1

	secret := 100
	shares, err := encrypt(int64(secret), k, n, int64(p))
	if err != nil {
		t.Fatal(err)
	}

	if len(shares) != n {
		t.Fatalf("len(shares) = %d, want %d", len(shares), n)
	}

	dec := decrypt(shares[:k], big.NewInt(int64(p)))
	if dec.Cmp(big.NewInt(int64(secret))) != 0 {
		t.Fatalf("dec = %d, want %d", dec, secret)
	}
}
