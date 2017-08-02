package com.xj.da.LSA

import edu.umd.cloud9.collection.XMLInputFormat
import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.umd.cloud9.collection.wikipedia.WikipediaPage
import edu.umd.cloud9.collection.wikipedia.language.EnglishWikipediaPage
import java.util.Properties

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source


/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/31
  */
object TFIDFDemo {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("tf-idf")
    val sc = new SparkContext(sparkConf)

    val path = "baidu.xml"
    @transient
    val conf = new Configuration()
    conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
    conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
    val kvs = sc.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf)
    val rawXmls = kvs.map(p => p._2.toString)
    // 分析准备数据
    val plainText: RDD[(String, String)] = rawXmls.flatMap(wikiXmlToPlainText)
    // 词形归并
    val stopWords = sc.broadcast(
      Source.fromFile("stopwords.txt").getLines().toSet
    ).value
    val lemmatized: RDD[Seq[String]] = plainText.mapPartitions(it => {
      val pipeline = createNLPPipeline()
      it.map{ case(title, contents) =>
        plainTextToLemmas(contents, stopWords, pipeline)
      }
    })
    // 计算TFIDF
    val docTermFreqs = lemmatized.map(terms => {
      val termFreqs = terms.foldLeft(new mutable.HashMap[String, Int]()){
        (map, term) => {
          map += term -> (map.getOrElse(term, 0) + 1)
          map
        }
      }
      termFreqs
    })
    docTermFreqs.cache()
  }

  /**
    * TF - IDF
    *
    * @param termFrequencyIntDoc
    * @param totalTermIntDoc
    * @param termFreqInCorpus
    * @param totalDocs
    * @return
    */
  def termDocWeight(termFrequencyIntDoc: Int, totalTermIntDoc: Int,
                    termFreqInCorpus: Int, totalDocs: Int): Double = {
    val tf: Double = termFrequencyIntDoc.toDouble / totalTermIntDoc
    val docFreq: Double = totalDocs.toDouble / termFreqInCorpus
    val idf: Double = math.log(docFreq)
    tf * idf
  }

  /**
    * 将维基百科的xml文件转化成纯文本
    *
    * @param xml
    * @return
    */
  def wikiXmlToPlainText(xml: String): Option[(String, String)] = {
    val page = new EnglishWikipediaPage()
    WikipediaPage.readPage(page, xml)
    if (page.isEmpty) None
    else Some((page.getTitle, page.getContent))
  }

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP): Seq[String] = {
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }

}