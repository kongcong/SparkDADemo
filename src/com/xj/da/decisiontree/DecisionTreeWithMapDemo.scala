package com.xj.da.decisiontree

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/18
  */
object DecisionTreeWithMapDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeWithMapDemo").setMaster("local")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("data/covtype.data")

    val data: RDD[LabeledPoint] = rawData.map { line =>
      val values: Array[Double] = line.split(",").map(_.toDouble)
      val wilderness: Double = values.slice(10, 14).indexOf(1.0).toDouble // wilderness对应的4个二元特征中哪一个取值为1
      val soil: Double = values.slice(14, 54).indexOf(1.0).toDouble // 对应40个二元特征
      val featureVector: linalg.Vector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil) // 将推导出的特征加回到前十个特征中
      val label: Double = values.last - 1
      LabeledPoint(label, featureVector)
    }
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val evaluations =
      for (impurity <- Array("gini", "entropy");
          depth <- Array(10, 20, 30);
          bins <- Array(40, 300))  // 三重for循环
        yield {
          val model: DecisionTreeModel = DecisionTree.trainClassifier(
            trainData, 7, Map(10 -> 4, 11 -> 40), impurity, depth, bins) // 指定类别型特征10和11的取值个数
          val trainAccuracy: Double = getMetrics(model, trainData).precision
          val cvAccuracy: Double = getMetrics(model, cvData).precision
          ((impurity, depth, bins), (trainAccuracy, cvAccuracy)) // 返回在训练集和CV集上的准确度
        }
    // 根据第二个值（准确度）降序排序并打印
    evaluations.sortBy(_._2).reverse.foreach(println)
    /**
      * ((gini,30,300),(0.9996643746611475,0.9359687333207073))
      * ((entropy,30,300),(0.9995955284377931,0.9427007110759112))
      * ((gini,30,40),(0.9995912255488335,0.9337476971815224))
      * ((entropy,30,40),(0.9995159249920397,0.9377765534339974))
      * ((gini,20,300),(0.9702304627326788,0.9250185086344932))
      * ((entropy,20,40),(0.9677154241357647,0.9218677364370448))
      * ((entropy,20,300),(0.9651530537602947,0.9225908644823608))
      * ((gini,20,40),(0.9640472112976652,0.9212995644014393))
      * ((gini,10,300),(0.7952922091892497,0.7927893803481345))
      * ((gini,10,40),(0.7907031781137855,0.7875553106867995))
      * ((entropy,10,40),(0.7826223526475676,0.7800485528830426))
      * ((entropy,10,300),(0.7781408937960947,0.7754515245949622))
      *
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
