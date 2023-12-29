package shamir

import (
	"errors"
	"math/big"
	"math/rand"
	"time"
)

func encrypt(secretInt int64, k int, n int, p int64) ([]int64, error) {
	// secretInt がint型であることをチェック
	if secretInt < 0 {
		return nil, errors.New("secretInt must be non-negative")
	}

	// 係数をランダムに決める
	rand.Seed(time.Now().UnixNano())
	a := make([]int64, k-1)
	for i := range a {
		a[i] = rand.Int63n(100-10+1) + 10 // 10から100の間のランダムな数
	}

	// n個のシェアを作成する
	shares := make([]int64, n)
	for i := 1; i <= n; i++ {
		var share int64 = 0
		for j := 1; j < k; j++ {
			share += a[j-1] * pow(int64(i), int64(j), p)
		}
		share += secretInt
		share %= p
		shares[i-1] = share
	}

	return shares, nil
}

// pow は累乗を計算し、結果をモジュロpで返す
func pow(base, exponent, mod int64) int64 {
	result := int64(1)
	for exponent > 0 {
		if exponent%2 == 1 {
			result = (result * base) % mod
		}
		base = (base * base) % mod
		exponent /= 2
	}
	return result
}

func lagrangeInterpreter(xList []int64, i int, p *big.Int) *big.Int {
	xI := big.NewInt(xList[i])
	res := big.NewInt(1)

	for cnt, xAtom := range xList {
		if cnt != i {
			numerator := new(big.Int).Neg(big.NewInt(xAtom))
			numerator.Add(numerator, p)
			numerator.Mod(numerator, p)

			denominator := new(big.Int).Sub(xI, big.NewInt(xAtom))
			denominator.Add(denominator, p)
			denominator.Mod(denominator, p)

			invDenominator := new(big.Int).ModInverse(denominator, p)
			res.Mul(res, numerator)
			res.Mul(res, invDenominator)
			res.Mod(res, p)
		}
	}
	return res
}

func lagrange(xList []int64, yList []int64, p *big.Int) *big.Int {
	res := big.NewInt(0)

	for n := range xList {
		term := lagrangeInterpreter(xList, n, p)
		term.Mul(term, big.NewInt(yList[n]))
		res.Add(res, term)
		res.Mod(res, p)
	}
	return res
}

func decrypt(shares []int64, p *big.Int) *big.Int {
	k := len(shares)
	xList := make([]int64, k)
	for i := range xList {
		xList[i] = int64(i + 1)
	}
	f0 := lagrange(xList, shares, p)

	if f0.Cmp(new(big.Int).Div(p, big.NewInt(2))) > 0 {
		f0.Sub(f0, p)
	}

	return f0
}
