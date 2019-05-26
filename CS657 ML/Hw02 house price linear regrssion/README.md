# HW02 
## Implement Linear Regression with SGD
> Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## SGD
> 一次跑一個樣本
### Flow
![](https://cdn-images-1.medium.com/max/800/1*cHd3oy9WcRh85H1EzNMsLA.png)

### Weights update
參數的更新公式
![](https://cdn-images-1.medium.com/max/800/1*_WPiHcBojP-u1pDh6sgExQ.png)

對weight/ m的summation的部分可以看成是內積的結果

由於SGD每次訓練時只採用一筆隨機挑選的資料，故上方公式的n=1