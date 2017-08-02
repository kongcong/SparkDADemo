package com.xj.da.decisiontree

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/18
  */
object RandomForestDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestDemo").setMaster("local")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("data/covtype.data")

    val data = rawData.map { line =>
      val values: Array[Double] = line.split(",").map(_.toDouble)
      val featureVector: linalg.Vector = Vectors.dense(values.init) // init返回除最后一个值之外的所有值，最后一列是目标
    val label: Double = values.last - 1 // 决策树要求label从0开始，所以要减1
      LabeledPoint(label, featureVector)
    }
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()


    // 训练模型
    val forestModel = RandomForest.trainClassifier(
      trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300)  // 20棵树  ‘auto’特征决策树每层的评估特征选择策略

    // 进行预测
    val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
    val vector: linalg.Vector = Vectors.dense(input.split(",").map(_.toDouble))
    val predict: Double = forestModel.predict(vector)

    println(predict)

  }
}
