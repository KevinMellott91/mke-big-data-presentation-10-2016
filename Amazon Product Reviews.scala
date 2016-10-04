// Databricks notebook source exported at Tue, 4 Oct 2016 18:47:51 UTC
// MAGIC %md
// MAGIC ##Amazon Product Reviews
// MAGIC Many interesting [public datasets](https://github.com/caesar0301/awesome-public-datasets) are available for download, including a sample of Amazon's product ratings. Databricks hosts a number of these datasets to help others get started with their experiments.
// MAGIC 
// MAGIC In this notebook, we will explore the Amazon product review data and show a couple of ways that insight can be gain during an initial analysis.

// COMMAND ----------

// MAGIC %md
// MAGIC ###Step 1: Load the Data
// MAGIC All of the datasets hosted by Databricks can be located by running the following command in a new cell. Alternatively, you can upload your own files into the Databricks File System (DBFS) or link to files stored on AWS S3.
// MAGIC 
// MAGIC ```display(dbutils.fs.ls("dbfs:/databricks-datasets/").toDF())```

// COMMAND ----------

// The product reviews are stored as Parquet files.
val productReviews = sqlContext.read.format("parquet").load("dbfs:/databricks-datasets/amazon/test4K").toDF().cache()
display(productReviews)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 2: Explore the Data
// MAGIC Sometimes you don't know which questions to ask until you have dug around a bit. In our case, let's start by examining the ratings being left by users.

// COMMAND ----------

import org.apache.spark.sql.functions.count

val ratings = productReviews.groupBy("rating").agg(count($"rating").alias("count")).orderBy($"count".desc)
display(ratings)

// COMMAND ----------

// MAGIC %md
// MAGIC Spark also provides some built-in functionality to help analyze numeric values.

// COMMAND ----------

display(productReviews.describe("rating"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 3: Clean/Enrich the Data
// MAGIC In this example, we are starting off with fairly clean data; however, that is most often not the case. For example, you will often find tabs, extra spaces, and line breaks contained within text you are attempting to examine.
// MAGIC 
// MAGIC For this exercise, we will attempt to enrich our data to contain some additional information.

// COMMAND ----------

// MAGIC %md
// MAGIC If our goal was to interpret each rating as being either "positive" or "negative", then given this information we may decide that a rating of 4/5 is positive and a rating of 1/2/3 is negative. We can append this information to our original DataFrame using the <i>withColumn</i> method in combination with a user-defined-function (UDF).

// COMMAND ----------

import org.apache.spark.sql.functions.udf

val interpretRatingUDF = udf((rating: Double) => if (rating >= 4D) true else false)
val interpretedReviews = productReviews.withColumn("isPositive", interpretRatingUDF($"rating"))

display(interpretedReviews)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Step 4: Learn from the Data
// MAGIC It may be interesting to know what makes a review positive or negative. Let's start by locating phrases that are commonly found within review text, which can be accomplished using Spark's machine learning (ML) libraries.
// MAGIC 
// MAGIC Most of the algorithms in Spark ML require numeric values as input. Therefore, we will setup a text processing pipeline to transform the original review text into a useable format.

// COMMAND ----------

// MAGIC %md
// MAGIC Before we can analyze the review text, it is important to clean it. This is typically an interative process that takes a significant amount of time and effort.
// MAGIC 
// MAGIC To be successful with this step in the process, it may help to brush up on your [regular expression](https://regex101.com/) skills.

// COMMAND ----------

import scala.util.matching.Regex

val pattern = "[!,?.]".r
val cleanReviewUDF = udf((review: String) => pattern.replaceAllIn(review, ""))
val cleanedData = interpretedReviews.withColumn("cleanReview", cleanReviewUDF($"review"))

// COMMAND ----------

// MAGIC %md
// MAGIC Now use the DataFrame containing the clean version of the review text to locate the most common phrases (n-grams).

// COMMAND ----------

import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, NGram}
import org.apache.spark.ml.Pipeline

val tokenizer = new Tokenizer().setInputCol("cleanReview").setOutputCol("tokens")
val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("significantTokens")
val ngram = new NGram().setInputCol("significantTokens").setOutputCol("ngrams").setN(3) // The "n" hyper-parameter controls whether we are using bi-grams, tri-grams, etc.

val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, ngram))
val model = pipeline.fit(cleanedData)
val transformedReviews = model.transform(cleanedData)

display(transformedReviews)

// COMMAND ----------

// MAGIC %md
// MAGIC Now that the ngrams have been identified, we can perform some aggregation tasks to determine the phrases that appear most often within both the positive and negative reviews.
// MAGIC 
// MAGIC The first step is to separate each ngram onto its own record, and associate it with the positive/negative indicator of the containing review.

// COMMAND ----------

val individualNGrams = transformedReviews.select("isPositive", "ngrams").flatMap(r => {
  val positiveIndicator = r.getBoolean(0)
  val ngrams = r.getAs[Stream[String]](1).toList
  ngrams.map(s => (positiveIndicator, s))
})
.toDF("isPositive", "ngram")

display(individualNGrams)

// COMMAND ----------

// MAGIC %md
// MAGIC We can now perform aggregation tasks to discover the most commonly used phrases in positive reviews.

// COMMAND ----------

val positivePhrases = individualNGrams.filter("isPositive = true").groupBy("ngram").agg(count($"ngram").alias("count")).orderBy($"count".desc)
display(positivePhrases)

// COMMAND ----------

// MAGIC %md
// MAGIC We can do the same analysis for negative reviews.

// COMMAND ----------

val negativePhrases = individualNGrams.filter("isPositive = false").groupBy("ngram").agg(count($"ngram").alias("count")).orderBy($"count".desc)
display(negativePhrases)

// COMMAND ----------

// MAGIC %md
// MAGIC ### A Step Further
// MAGIC Now that we can associate short phrases with positive or negative reviews, we could choose to implement a sentiment analysis engine built from the context of product reviews. Given new feedback about a product, we would then be able to predict if the user input was positive or negative.
