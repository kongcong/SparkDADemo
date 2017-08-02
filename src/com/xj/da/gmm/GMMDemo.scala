package com.xj.da.gmm

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/19
  */
object GMMDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("GMMDemo")
    val sc = new SparkContext(conf)

    val rawData: RDD[String] = sc.textFile("data/3165.csv")
    // println(rawData.count())

    val finaltorqueData: RDD[linalg.Vector] = rawData.map { line =>
      Vectors.dense(line.split(",").map(_.toDouble))
    }
    //finaltorque.foreach(println)

    val Array(trainData, testData) = finaltorqueData.randomSplit(Array(0.8, 0.2))

    val gaussianMixture = new GaussianMixture()
    //val initModel = new GaussianMixtureModel(Array(), Array())
    val mixtureModel = gaussianMixture
      //.setInitialModel(initModel)
      .setK(2)
      .setConvergenceTol(0.0001)
      .run(trainData)


    val predict: RDD[Int] = mixtureModel.predict(testData)
    testData.zip(predict).saveAsTextFile("out/gmm")

    for (i <- 0 until mixtureModel.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (mixtureModel.weights(i), mixtureModel.gaussians(i).mu, mixtureModel.gaussians(i).sigma))
    }

  }
}
