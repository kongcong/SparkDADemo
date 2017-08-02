package com.xj.da.kmeans

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/18
  */
object KMeansDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("KMeansDemo")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("data/kddcup.data_10_percent_corrected")
    //println(rawData.count()) // 493672

    // 分类统计样本个数，按照样本从多到少排序
    // rawData.map(_.split(",").last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)
    /**
      * (smurf.,280631)
      * (neptune.,107083)
      * (normal.,97206)
      * (back.,2203)
      * (satan.,1589)
      * (ipsweep.,1247)
      * (portsweep.,1040)
      * (warezclient.,1020)
      * (teardrop.,979)
      * (pod.,264)
      * (nmap.,231)
      * (guess_passwd.,53)
      * (buffer_overflow.,30)
      * (land.,21)
      * (warezmaster.,20)
      * (imap.,12)
      * (rootkit.,10)
      * (loadmodule.,9)
      * (ftp_write.,8)
      * (multihop.,7)
      * (phf.,4)
      * (perl.,3)
      * (spy.,2)
      */

    val labelsAndData: RDD[(String, linalg.Vector)] = rawData.map { line =>
      val buffer = line.split(',').toBuffer // 创建一个buffer,他是一个可变列表
      buffer.remove(1, 3)
      val label: String = buffer.remove(buffer.length - 1)
      val vector: linalg.Vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
    val data: RDD[linalg.Vector] = labelsAndData.values.cache()

    // 聚类
    //val model: KMeansModel = new KMeans().run(data)
    //model.clusterCenters.foreach(println)
    /**
      * 输出两个向量，代表KMeans将数据聚成两类，因为类型有23个 所以肯定没有准确刻画数据中的不同群组
      * [48.01307145852197,1623.0182165855397,867.8036992247874,4.456409228008127E-5,0.006437485693913557,1.4179483907298585E-5,0.03454122279817935,1.5192304186391342E-4,0.14820801708020118,0.01021935661604591,1.1141023070020317E-4,3.646153004733922E-5,0.011359792250304352,0.001083717698629249,1.0938459014201766E-4,0.001008768997976385,0.0,0.0,0.001387563782357076,332.29900075151266,292.94576752533567,0.1765699423300134,0.17649317055285815,0.05747370212145333,0.05775895282485708,0.7916297088546188,0.020981929260580873,0.02900113638435369,232.48885593846913,188.67760107439975,0.7538364619352064,0.0309099177387495,0.6020196041493157,0.0066778482025469385,0.17663944205756066,0.1763269667450554,0.05815806478403588,0.05745170366499057]
      * [2.0,6.9337564E8,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,57.0,3.0,0.79,0.67,0.21,0.33,0.05,0.39,0.0,255.0,3.0,0.01,0.09,0.22,0.0,0.18,0.67,0.05,0.33]
      *
      */


    /**
      * 利用前面得到的KMeans模型，下面的代码先为每个数据点分配一个簇
      * 然后对簇-类别对进行计数并以可读的方式输出
      */
//    val clusterLabelCount: collection.Map[(Int, String), Long] = labelsAndData.map { case (label, datum) =>
//      val cluster = model.predict(datum)
//      (cluster, label)
//    }.countByValue()
//    clusterLabelCount.toSeq.sorted.foreach {
//      case ((cluster, label), count) =>
//        println(f"$cluster%1s$label%18s$count%8s")  // 使用字符串插值器对变量的输出进行格式化
//    }
    /**
      * 0             back.    2203
      * 0  buffer_overflow.      30
      * 0        ftp_write.       8
      * 0     guess_passwd.      53
      * 0             imap.      12
      * 0          ipsweep.    1247
      * 0             land.      21
      * 0       loadmodule.       9
      * 0         multihop.       7
      * 0          neptune.  107083
      * 0             nmap.     231
      * 0           normal.   97206
      * 0             perl.       3
      * 0              phf.       4
      * 0              pod.     264
      * 0        portsweep.    1039
      * 0          rootkit.      10
      * 0            satan.    1589
      * 0            smurf.  280631
      * 0              spy.       2
      * 0         teardrop.     979
      * 0      warezclient.    1020
      * 0      warezmaster.      20
      * 1        portsweep.       1
      * 结果显示聚类根本没有任何作用。簇1只有1个数据点
      */

    // 对k的取值进行评价 比如5~40
    //(5 to 40 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)
    /**
      * (5,1748.4253942444539)
      * (10,1269.5692402883699)
      * (15,719.6050228994742)
      * (20,526.860122356017)
      * (25,412.854554256069)
      * (30,336.44742196670506)
      * (35,314.8603062163254)
      * (40,290.91329753423656)
      */

    val kmeans = new KMeans()
    // 设置在给定k值时运行的次数
    //kmeans.setRuns(10)
    // 阈值 控制聚类过程中簇质心进行有效移动的最小值
    //kmeans.setEpsilon(1.0e-6) // 默认为1.0e-4,这里比默认值小
    //val kMeansModel = kmeans.run(data)
    //(30 to 100 by 10).par.map(k => (k, clusteringScore(data, k))).toList.foreach(println)
    /**
      * (30,372.1636931723143)
      * (40,256.79393582909194)
      * (50,154.78659076305547)
      * (60,243.1443074917606)
      * (70,112.49933125931568)
      * (80,112.60539203261114)
      * (90,100.11313726285633)
      * (100,94.90984248130513)
      */

    /**
      * 我们需要找到一个临界点，过了这个临界点之后继续增加k值并不会显著得降低得分，这个点就是k值-得分曲线的拐点
      * 这条曲线通常在拐点之后会继续下行但最终趋于水平
      *
      * 本示例中，在k值过了100之后得分下降还是很明显，所以k的拐点值应该大于100
      */

    // 使用k=100构建一个模型，并把每个数据点映射到一个簇编号，保存特征向量
//    val kMeansModelWithKeq100: KMeansModel = kmeans.setK(100).run(data)
//    val sample = data.map(datum =>
//      kMeansModelWithKeq100.predict(datum) + "," + datum.toArray.mkString(",") // 用分隔符把集合元素连接成一个字符串
//    ).sample(false, 0.05)

    // sample.saveAsTextFile("out/kmeans/sample")

    val dataAsArray: RDD[Array[Double]] = data.map(_.toArray)
    val numCols: Int = dataAsArray.first().length
    val n = dataAsArray.count()
    val sums = dataAsArray.reduce(
      (a, b) => a.zip(b).map(t => t._1 * t._2))
    val sumSquares = dataAsArray.fold( // 把平方和汇总到一个初值为0的数组中
      new Array[Double](numCols)
    )(
      (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2)
    )
    val stdevs = sumSquares.zip(sums).map {
      case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
    }
    val means: Array[Double] = sums.map(_ / n)

    /**
      * 将每个特征转化成标准得分
      * @param datum
      * @return
      */
    def normalize(datum: linalg.Vector) = {
      val normalizedArray: Array[Double] = (datum.toArray, means, stdevs).zipped.map(
        (value, mean, stdev) =>
          if (stdev <= 0)(value - mean) else (value - mean) / stdev
      )
      Vectors.dense(normalizedArray)
    }

    // 增加k的取值范围并在规范化的数据上运行相同的测试
    val normalizeData: RDD[linalg.Vector] = data.map(normalize)

    (60 to 120 by 10).par.map(k =>
      (k, clusteringScore(normalizeData, k))).toList.foreach(println)
  }

  /**
    * 计算欧式距离 两个向量相应元素的差的平方的和的平方根
    * @param a
    * @param b
    * @return
    */
  def distance(a: linalg.Vector, b: linalg.Vector): Double = {
    math.sqrt(a.toArray.zip(b.toArray)
      .map(p => p._1 - p._2)
      .map(d => d * d).sum)
  }

  /**
    * 计算数据点到最近簇质心距离
    * @param dataum
    * @param model
    * @return
    */
  def distToCentroid(dataum: linalg.Vector, model: KMeansModel): Double = {
    val cluster: Int = model.predict(dataum)
    val centroid: linalg.Vector = model.clusterCenters(cluster)
    distance(centroid, dataum)
  }

  /**
    * 给定k值的模型计算平均质心距离函数
    * @param data
    * @param k
    * @return
    */
  def clusteringScore(data: RDD[linalg.Vector], k: Int) = {
    val kmeans: KMeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }

}
