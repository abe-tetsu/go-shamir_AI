package prediction

import (
	"golang.org/x/xerrors"
)

func Predict(x []float64, weights [][]float64, biases []float64) ([]float64, error) {
	// 入力ベクトルの長さを確認
	if len(x) != 784 {
		return nil, xerrors.Errorf("Input vector must have a length of 784, got %d", len(x))
	}

	// 重み行列の寸法を確認
	if len(weights) != 784 || len(weights[0]) != 10 {
		return nil, xerrors.Errorf("Weights must be a 784x10 dimensional matrix, got %d x %d", len(weights), len(weights[0]))
	}

	// バイアスの長さを確認
	if len(biases) != 10 {
		return nil, xerrors.Errorf("Biases must have a length of 10, got %d", len(biases))
	}

	// 出力値を格納する配列を初期化（10個の出力ノードに対応）
	outputValues := make([]float64, 10)

	// 各出力ノードに対して線形結合を計算
	for i := 0; i < 10; i++ {
		// 重みと入力の積の合計を求める
		sumWeightedInputs := 0.0
		for j := 0; j < 784; j++ {
			sumWeightedInputs += x[j] * weights[j][i]
		}
		// バイアスを追加
		sumWeightedInputs += biases[i]
		// 計算された値を出力値に設定
		outputValues[i] = sumWeightedInputs
	}

	return outputValues, nil
}
