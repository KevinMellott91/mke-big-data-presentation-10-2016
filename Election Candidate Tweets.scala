// Databricks notebook source exported at Tue, 4 Oct 2016 18:49:24 UTC
// MAGIC %md
// MAGIC ## Election Candidate Tweets
// MAGIC Kaggle [hosts a dataset](https://www.kaggle.com/benhamner/clinton-trump-tweets) containing the 3000 most recent tweets associated with either Hillary Clinton or Donald Trump. Seeing as we are nearing a presidental election, it may be a good time to poke around and see what kinds of things they are saying.
// MAGIC 
// MAGIC This notebook will use Spark ML and [Soundcloud's implementation](https://github.com/soundcloud/cosine-lsh-join-spark) of approximate nearest neighbors (ANN) to locate similar tweets and identify common themes being discussed. Before you can run this notebook, you will need to install the maven library using the following coordinates.
// MAGIC 
// MAGIC ```com.soundcloud:cosine-lsh-join-spark_2.10:0.0.4```

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 1: Connect to S3
// MAGIC The tweets have been uploaded to Amazon S3, so we will need to specify the credentials to access the file.

// COMMAND ----------

val AccessKey = "AKIAJHYM7LPX6OZFEYQA"
val SecretKey = "/lGuRQrhVj43Yg9iRbBABPZyHNFbF4Ip8lvDaBmo"
val EncodedSecretKey = SecretKey.replace("/", "%2F")
val AwsBucketName = "bdug"
val MountName = "s3"

// COMMAND ----------

// MAGIC %md
// MAGIC The next step will establish a local mount for the S3 storage bucket. Since the S3 bucket credentials are associated with a mount, we will begin by deleting any existing mount setups for our storage bucket.

// COMMAND ----------

dbutils.fs.unmount("/mnt/s3")
dbutils.fs.mount(s"s3a://$AccessKey:$EncodedSecretKey@$AwsBucketName", s"/mnt/$MountName")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 2: Parse the Data
// MAGIC Now we can use the local mount to access the data in S3. Since Amazon charges fees based on the number of file requests, it is important to use Spark's <i>cache</i> method.
// MAGIC 
// MAGIC In the previous exercise, we explored data using Spark's DataFrame API. This time, we will be using Spark SQL as an alternative approach. The ever-growing list of Spark SQL functions can be found in the [Spark code on Github](https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/functions.scala).
// MAGIC 
// MAGIC Once a DataFrame object has been created, we can simply invoke the <i>registerTempTable</i> method to make a Spark SQL table available. Note that this approach operates on in-memory data and will not persist when the Spark context is disposed.

// COMMAND ----------

sqlContext.read.format("com.databricks.spark.csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("escape", "\"")
  .option("parserLib", "univocity")
  .load(s"/mnt/$MountName/tweets.csv")
  .cache()
  .registerTempTable("tweets")

// COMMAND ----------

// MAGIC %md
// MAGIC Since a Spark SQL table alias has been established, we can now use SQL syntax to query the data.

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from tweets

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 3: Explore the Data
// MAGIC In the last exercise, we performed all of our analysis using Spark DataFrames. There is nothing wrong with this approach; however, it does require the user to gain information about the available [DataFrame API methods](http://spark.apache.org/docs/1.6.2/api/scala/index.html#org.apache.spark.sql.DataFrame).
// MAGIC 
// MAGIC For this exercise, we will use [Spark SQL](http://spark.apache.org/docs/1.6.2/sql-programming-guide.html#sql) instead. This query language supports syntax dating back to SQL 2003 and can be an easier way to get a wider variety of analysts started with Spark.
// MAGIC 
// MAGIC Let's begin by evaluating each candidates online presence. We'll start by viewing the popularity of each candidates' tweets.

// COMMAND ----------

// Provides access to the built-in SQL functions, such as "avg" and "max".
import org.apache.spark.sql.functions._

// COMMAND ----------

// MAGIC %sql
// MAGIC select handle as candidate, avg(retweet_count) as `re-tweets`, avg(favorite_count) as favorites, count(*) as `# of tweets` from tweets where is_retweet = 'False' group by handle

// COMMAND ----------

// MAGIC %md
// MAGIC Just for fun, let's see the top 5 most popular tweets from each candidate. We'll start with Hillary Clinton first.

// COMMAND ----------

// MAGIC %sql
// MAGIC select text as tweet, (favorite_count + retweet_count) as popularity from tweets where handle = 'HillaryClinton' order by (favorite_count + retweet_count) desc limit 5

// COMMAND ----------

// MAGIC %md
// MAGIC Now for the most popular tweets from Donald Trump.

// COMMAND ----------

// MAGIC %sql
// MAGIC select text as tweet, (favorite_count + retweet_count) as popularity from tweets where handle = 'realDonaldTrump' order by (favorite_count + retweet_count) desc limit 5

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 4: Phrase Detection via N-Grams
// MAGIC Pretty interesting tweets to say the least, but what are the overall messages each candidate is sending? We can use n-grams to recognize the most common phrases used within their tweets.
// MAGIC 
// MAGIC Since the return value of a Spark SQL query is a DataFrame, we can feed this information into a machine learning pipeline.

// COMMAND ----------

import scala.util.matching.Regex

val pattern = "[!,?.\"]".r
val cleanTweetUDF = udf((tweet: String) => pattern.replaceAllIn(tweet, ""))

// COMMAND ----------

val hillaryTweets = sqlContext.sql("select text as tweet, time from tweets where handle = 'HillaryClinton'").withColumn("cleanTweet", cleanTweetUDF($"tweet"))
val donaldTweets = sqlContext.sql("select text as tweet, time from tweets where handle = 'realDonaldTrump'").withColumn("cleanTweet", cleanTweetUDF($"tweet"))

// COMMAND ----------

// MAGIC %md
// MAGIC Now we will construct a machine learning pipeline to process the n-grams within the tweets.

// COMMAND ----------

import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, NGram}
import org.apache.spark.ml.Pipeline

val tokenizer = new Tokenizer().setInputCol("cleanTweet").setOutputCol("tokens")
val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("significantTokens")
val ngram = new NGram().setInputCol("significantTokens").setOutputCol("ngrams").setN(3) // The "n" hyper-parameter controls whether we are using bi-grams, tri-grams, etc.

val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, ngram))

// COMMAND ----------

// MAGIC %md
// MAGIC Let's process Hillary Clinton's tweets first, using the ML pipeline above. Notice how the data changes with each stage of the pipeline.

// COMMAND ----------

val hillaryModel = pipeline.fit(hillaryTweets)
val hillaryNGrams = hillaryModel.transform(hillaryTweets)

hillaryNGrams.registerTempTable("hillaryNGrams")
display(hillaryNGrams)

// COMMAND ----------

// MAGIC %md
// MAGIC We can aggregate the results using Spark SQL to view the most commonly used phrases.

// COMMAND ----------

sqlContext.sql("select explode(ngrams) as ngram from hillaryNGrams").registerTempTable("flattenedHillaryNGrams")

// COMMAND ----------

// MAGIC %sql
// MAGIC select ngram as phrase, count(*) as `# times` from flattenedHillaryNGrams group by ngram order by count(*) desc limit 10

// COMMAND ----------

// MAGIC %md
// MAGIC We will now perform the same analysis for Donald Trump, condensed into a couple of cells. Notice that we are simply using the same ML pipeline with different data.

// COMMAND ----------

val donaldModel = pipeline.fit(donaldTweets)
val donaldNGrams = donaldModel.transform(donaldTweets)

donaldNGrams.registerTempTable("donaldNGrams")
sqlContext.sql("select explode(ngrams) as ngram from donaldNGrams").registerTempTable("flattenedDonaldNGrams")

// COMMAND ----------

// MAGIC %sql
// MAGIC select ngram as phrase, count(*) as `# times` from flattenedDonaldNGrams group by ngram order by count(*) desc limit 10

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 5: Similar Messages via Approximate Nearest Neighbors (ANN)
// MAGIC Our phrase detection analysis shed some light on the primary messages being sent by each candidate. Name-calling aside, are there any issues or topics that they agree upon?
// MAGIC 
// MAGIC Looking for similarities in the N-Gram patterns could help us answer this question, but we are going to take a different approach. Instead, we will use term frequency (TF) and inverse document frequency (IDF) to transform each tweet into a numeric feature vector. We will then use Soundcloud's implementation of ANN to compare the tweets to one another and check for similarities.
// MAGIC 
// MAGIC The first thing we will need to do is modify our pipeline to contain the [HashingTF](http://spark.apache.org/docs/1.6.2/api/scala/index.html#org.apache.spark.ml.feature.HashingTF) and [IDF](http://spark.apache.org/docs/1.6.2/api/scala/index.html#org.apache.spark.ml.feature.IDF) pipeline components.

// COMMAND ----------

import org.apache.spark.ml.feature.{HashingTF, IDF}

val hashingTF = new HashingTF().setInputCol("significantTokens").setOutputCol("hashedTokens")
val idf = new IDF().setInputCol("hashedTokens").setOutputCol("features")

val similaritiesPipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))

// COMMAND ----------

// MAGIC %md
// MAGIC Now that the pipeline has been established, let's process the tweets from each candidate. Once again, look at how the data is transformed as it makes its way through the pipeline.

// COMMAND ----------

val hillarySimilarModel = similaritiesPipeline.fit(hillaryTweets)
val hillarySimilarFeatures = hillarySimilarModel.transform(hillaryTweets)

val donaldSimilarModel = similaritiesPipeline.fit(donaldTweets)
val donaldSimilarFeatures = donaldSimilarModel.transform(donaldTweets)

display(hillarySimilarFeatures)

// COMMAND ----------

// MAGIC %md
// MAGIC The Approximate Nearest Neighbors algorithm requires each record to have a numeric identifier, so we will need to massage our data a bit. We can call Spark's <i>zipWithIndex</i> method to fill this need.

// COMMAND ----------

import org.apache.spark.mllib.linalg.SparseVector

val hillaryWithID = hillarySimilarFeatures.select("tweet", "time", "features").map(t => (t.getString(0), t.getString(1), t.getAs[SparseVector](2))).zipWithIndex().map(t => {
  (t._2, t._1._1, t._1._2, t._1._3)
}).toDF("id", "tweet", "time", "features")
hillaryWithID.registerTempTable("hillaryWithID")

val donaldWithID = donaldSimilarFeatures.select("tweet", "time", "features").map(t => (t.getString(0), t.getString(1), t.getAs[SparseVector](2))).zipWithIndex().map(t => {
  (t._2, t._1._1, t._1._2, t._1._3)
}).toDF("id", "tweet", "time", "features")
donaldWithID.registerTempTable("donaldWithID")

display(hillaryWithID)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can use ANN to calculate the similarities between tweets. We will first analyze each candidate separately, to check for duplicate messaging patterns.

// COMMAND ----------

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import com.soundcloud.lsh.{ Cosine, Lsh }
import org.apache.spark.storage.StorageLevel

val numPartitions = 3
val storageLevel = StorageLevel.MEMORY_AND_DISK

val lsh = new Lsh(
  minCosineSimilarity = 0.75,
  dimensions = 20,
  numNeighbours = 200,
  numPermutations = 10,
  partitions = numPartitions,
  storageLevel = storageLevel
)

val hillaryInputRows = hillaryWithID.map(r => IndexedRow(r.getLong(0), r.getAs[SparseVector](3)))
val hillaryMatrix = new IndexedRowMatrix(hillaryInputRows)
val hillarySimilarityMatrix = lsh.join(hillaryMatrix)

val donaldInputRows = donaldWithID.map(r => IndexedRow(r.getLong(0), r.getAs[SparseVector](3)))
val donaldMatrix = new IndexedRowMatrix(donaldInputRows)
val donaldSimilarityMatrix = lsh.join(donaldMatrix)

// COMMAND ----------

// MAGIC %md
// MAGIC The resulting data contains a relationship between two tweets, and a similarity score between 0 and 1.

// COMMAND ----------

hillarySimilarityMatrix.entries.toDF("tweetIdA", "tweetIdB", "score").registerTempTable("hillarySimilar")
donaldSimilarityMatrix.entries.toDF("tweetIdA", "tweetIdB", "score").registerTempTable("donaldSimilar")

// COMMAND ----------

// MAGIC %md
// MAGIC We can join this information back to the data containing the actual tweets to see the results.

// COMMAND ----------

// MAGIC %sql
// MAGIC select t1.time as time1, t1.tweet as tweet1, t2.time as time2, t2.tweet as tweet2, s.score from hillarySimilar as s join hillaryWithID as t1 on t1.id = s.tweetIdA join hillaryWithID as t2 on t2.id = s.tweetIdB order by s.score desc

// COMMAND ----------

// MAGIC %md
// MAGIC Now for Donald's tweets...

// COMMAND ----------

// MAGIC %sql
// MAGIC select t1.time as time1, t1.tweet as tweet1, t2.time as time2, t2.tweet as tweet2, s.score from donaldSimilar as s join donaldWithID as t1 on t1.id = s.tweetIdA join donaldWithID as t2 on t2.id = s.tweetIdB order by s.score desc

// COMMAND ----------

// MAGIC %md
// MAGIC Now let's analyze all of the tweets as a single set and find out if the candidates are talking about similar things.

// COMMAND ----------

val allTweets = hillarySimilarFeatures.select("tweet", "time", "features").map(t => ("HillaryClinton", t.getString(0), t.getString(1), t.getAs[SparseVector](2)))
  .union(
    donaldSimilarFeatures.select("tweet", "time", "features").map(t => ("realDonaldTrump", t.getString(0), t.getString(1), t.getAs[SparseVector](2)))
  ).zipWithIndex().map(t => {
    (t._2, t._1._1, t._1._2, t._1._3, t._1._4)
  }).toDF("id", "candidate", "tweet", "time", "features")
allTweets.registerTempTable("allTweets")

val allInputRows = allTweets.map(r => IndexedRow(r.getLong(0), r.getAs[SparseVector](4)))
val allMatrix = new IndexedRowMatrix(allInputRows)
val allSimilarityMatrix = lsh.join(allMatrix)

allSimilarityMatrix.entries.toDF("tweetIdA", "tweetIdB", "score").registerTempTable("allSimilar")

// COMMAND ----------

// MAGIC %md
// MAGIC Based on what we've already uncovered, it's no big surprise that Donald Trump and Hillary Clinton don't tweet about the same things!

// COMMAND ----------

// MAGIC %sql
// MAGIC select t1.candidate, t1.tweet as tweet1, t2.candidate, t2.tweet as tweet2, s.score from allSimilar as s join allTweets as t1 on t1.id = s.tweetIdA join allTweets as t2 on t2.id = s.tweetIdB where t1.candidate <> t2.candidate order by s.score desc
