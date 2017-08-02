package com.xj.da.gmm

import breeze.numerics.sqrt
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/19
  */
object GMMWithOneVariable {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      //.setMaster("local")
      .setAppName("GMMWithOneVariable")
    val sc = new SparkContext(conf)

    //val rawData: RDD[String] = sc.textFile("data/resv.csv")
    val rawData: RDD[String] = sc.textFile("hdfs://master:8020/home/kongc/data/3165-1.csv")
    val count = rawData.count()
    println("count: " + count)

    val finaltorqueData: RDD[linalg.Vector] = rawData.map { line =>
      Vectors.dense(line.split(",")(0).toDouble,line.split(",")(2).toDouble)
    }

    val finaltorqueData1: RDD[linalg.Vector] = finaltorqueData.map(f => Vectors.dense(f.toArray(0)))
    finaltorqueData1.foreach(println)

    val Array(trainData, testData) = finaltorqueData1.randomSplit(Array(0.8, 0.2))

    // 指定初始模型(GMM算法受初始值指定影响很大)
    // 0
    val filter0: RDD[linalg.Vector] = finaltorqueData.filter(_.toArray(1).toInt == 0)
    // 非0
    val filter1: RDD[linalg.Vector] = finaltorqueData.filter(_.toArray(1).toInt != 0)
    // 权重
    println("f0:  " + filter0.count())
    val w1: Double = (filter0.count()/count.toDouble)
    val w2: Double = (filter1.count()/count.toDouble)
    println(s"w1 = $w1")
    // 均值
    val m1: Double = filter0.map(_.toArray(0)).mean()
    val m2: Double = filter1.map(_.toArray(0)).mean()
    // 标准差
    val v1: Double = sqrt(filter0.map(_.toArray(0)).variance())
    val v2: Double = sqrt(filter1.map(_.toArray(0)).variance())
    val mu1: linalg.Vector = Vectors.dense(Array(m1))
    val mu2: linalg.Vector = Vectors.dense(Array(m2))
    println(s"mu1 : $mu1")
    println(s"mu2 : $mu2")

    val sigma1: Matrix = Matrices.dense(1, 1, Array(filter0.map(_.toArray(0)).variance()))
    val sigma2: Matrix = Matrices.dense(1, 1, Array(filter1.map(_.toArray(0)).variance()))
    //println(sigma.numRows + "::" + sigma.numCols)
    // 构建一个MultivariateGaussian需要两个参数 一个是均值向量 一个是协方差矩阵
    val gmm1 = new MultivariateGaussian(mu1, sigma1)
    val gmm2 = new MultivariateGaussian(mu2, sigma2)
    val gaussians = Array(gmm1, gmm2)
    // 构建一个GaussianMixtureModel需要两个参数 一个是权重数组 一个是组成混合高斯分布的每个高斯分布
    val initModel = new GaussianMixtureModel(Array(w1, w2), gaussians)
    for (i <- 0 until initModel.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (initModel.weights(i), initModel.gaussians(i).mu, initModel.gaussians(i).sigma))
    }

    val gaussianMixture = new GaussianMixture()
    val mixtureModel = gaussianMixture
      .setInitialModel(initModel)
      .setK(2)
      .setConvergenceTol(0.0001)
      .run(finaltorqueData1)

    val predict: RDD[Int] = mixtureModel.predict(finaltorqueData1)
    rawData.zip(predict).saveAsTextFile("hdfs://master:8020/home/kongc/data/out/gmm/3165-13")

    for (i <- 0 until mixtureModel.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (mixtureModel.weights(i), mixtureModel.gaussians(i).mu, mixtureModel.gaussians(i).sigma))
    }
    //mixtureModel.save(sc, "out/gmm")
  }
}
