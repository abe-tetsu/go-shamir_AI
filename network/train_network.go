package network

import (
	"fmt"
	"github.com/abe-tetsu/shamir-ai/util"
)

type NeuralNetwork struct {
	Input        int64
	Output       int64
	Epoch        int64
	LearningRate float64
	Weight       [][]float64
	Bias         []float64
}

func NewNeuralNetwork(input, output, epoch int64, learningRate float64) *NeuralNetwork {
	nn := NeuralNetwork{Input: input, Output: output, Epoch: epoch, LearningRate: learningRate}
	return &nn
}

func (nn *NeuralNetwork) TrainNetWork(trainDataSet *util.MnistDataSet) (*util.MnistDataSet, error) {
	// data数
	dataNum := len(trainDataSet.DataSet)
	fmt.Println("学習データ数:", dataNum)

	// weightとbiasの初期化
	weight := make([][]float64, nn.Input)
	for i := range weight {
		weight[i] = make([]float64, nn.Output)
		for j := range weight[i] {
			weight[i][j] = 0.0
		}
	}

	bias := make([]float64, nn.Output)
	for i := range bias {
		bias[i] = 0
	}

	// 学習
	for i := int64(0); i < nn.Epoch; i++ {
		for _, data := range trainDataSet.DataSet {
			dataImage := util.TransformData(data)

			// 順伝播の計算
			z := make([]float64, nn.Output)
			for j := range weight {
				dotProduct, err := util.Dot(dataImage, weight[j]) // エラーハンドリングは省略
				if err != nil {
					return nil, err
				}
				z[j] = dotProduct + bias[j]
			}
			a := util.Relu(z)

			// 誤差の計算
			dz := make([]float64, nn.Output)
			for j := range dz {
				dz[j] = a[j] - float64(data.Label)
			}

			// dwとdbの計算
			dw := util.Outer(dataImage, dz)
			db := dz

			// 重みとバイアスの更新
			for j := range weight {
				for k := range weight[j] {
					weight[j][k] -= nn.LearningRate * dw[j][k]
				}
			}
			for j := range bias {
				bias[j] -= nn.LearningRate * db[j]
			}
		}
		fmt.Printf("Epoch %d/%d\n", i+1, nn.Epoch)
	}

	nn.Weight = weight
	nn.Bias = bias

	// 重みとバイアスを保存
	err := util.SaveWeights(weight, "go_ai/ai-data/weights.gob")
	if err != nil {
		return nil, err
	}

	err = util.SaveBias(bias, "go_ai/ai-data/bias.gob")
	if err != nil {
		return nil, err
	}

	return trainDataSet, nil
}
