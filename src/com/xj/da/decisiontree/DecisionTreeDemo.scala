package com.xj.da.decisiontree

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/17
  */
object DecisionTreeDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeDemo").setMaster("local")
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

    // 参数：训练集 分类数 类别型特征的信息 不纯度gini 最大深度 最大桶数
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)

    val metrics: MulticlassMetrics = getMetrics(model, cvData)

    /**
      * 混淆矩阵
      * 因为目标类别的取值有7个，所以混淆矩阵是一个7*7的矩阵，
      * 矩阵每一行对应一个实际的正确类别值，矩阵每一列按顺序对应预测值。
      *
      */
    val confusionMatrix: Matrix = metrics.confusionMatrix
    println(confusionMatrix)
    /**
      * 14329.0  6578.0   8.0     0.0    0.0  0.0   387.0
      * 5506.0   22354.0  437.0   16.0   0.0  2.0   52.0
      * 0.0      425.0    3086.0  57.0   0.0  15.0  0.0
      * 0.0      0.0      160.0   119.0  0.0  0.0   0.0
      * 0.0      913.0    44.0    1.0    0.0  0.0   0.0
      * 0.0      472.0    1149.0  33.0   0.0  67.0  0.0
      * 1156.0   31.0     0.0     0.0    0.0  0.0   875.0
      */

    // 准确度
    val precision = metrics.precision
    println(s"precision = $precision")  // precision = 0.6929612905449657 约有69%的预测正确

    // 计算每个类别相对于其他类别的精确度
    (0 until 7).map(  // DecisionTreeModel模型的类别标号从0开始
      cat => (metrics.precision(cat), metrics.recall(cat))
    ).foreach(println)
    /**
      * (0.6813197157245728,0.6804870023033893)
      * (0.7253870681437166,0.7825454803361821)
      * (0.6404286008654441,0.8454842219804135)
      * (0.5638766519823789,0.43986254295532645)to driver
      * (0.0,0.0)
      * (0.6923076923076923,0.03137710633352702)
      * (0.68,0.4425343811394892)
      */

    // 决策树调优
    val evaluations =
      for (impurity <- Array("gini", "entropy");
          depth <- Array(1, 20);
          bins <- Array(10, 300)
      ) yield {
        val model: DecisionTreeModel = DecisionTree.trainClassifier(
          trainData, 7, Map[Int, Int](), impurity, depth, bins)
        val predictionsAndLabels = cvData.map(example =>
          (model.predict(example.features), example.label)
        )
        val accuracy: Double = new MulticlassMetrics(predictionsAndLabels).precision
        ((impurity, depth, bins), accuracy)
      }

    // 最佳选择：不纯性度量采用熵，最大深度为20，桶数为300 这是得到的准确度是0.91
    evaluations.sortBy(_._2).reverse.foreach(println)
    /**
      * ((entropy,20,300),0.9126320498262276)
      * ((gini,20,300),0.9034100684766525)
      * ((entropy,20,10),0.896287120195451)
      * ((gini,20,10),0.8899212002339906)
      * ((gini,1,300),0.6360414300953168)
      * ((gini,1,10),0.6360070197171467)
      * ((entropy,1,300),0.488334881800351)
      * ((entropy,1,10),0.488334881800351)
      */
  }

  /**
    * 使用cv集计算模型的指标
    * @param model
    * @param data
    * @return
    */
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels: RDD[(Double, Double)] = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }
}
