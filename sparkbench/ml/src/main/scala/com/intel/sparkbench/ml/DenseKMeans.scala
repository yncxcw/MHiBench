/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.hibench.sparkbench.ml

import org.apache.hadoop.io.LongWritable
import org.apache.log4j.{Level, Logger}
import org.apache.mahout.math.VectorWritable
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
 *
 * An example k-means app. Run with
 * {{{
 * ./bin/run-example org.apache.spark.examples.mllib.DenseKMeans [options] <input>
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object DenseKMeans {

  object InitializationMode extends Enumeration {
    type InitializationMode = Value
    val Random, Parallel = Value
  }

  import com.intel.hibench.sparkbench.ml.DenseKMeans.InitializationMode._

  case class Params(
      input: String = null,
      k: Int = -1,
      storage: Int=3,
      numIterations: Int = 10,
      initializationMode: InitializationMode = Parallel)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("DenseKMeans") {
      head("DenseKMeans: an example k-means app for dense data.")
      opt[Int]('k', "k")
        .required()
        .text(s"number of clusters, required")
        .action((x, c) => c.copy(k = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default; ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("storage")
        .text(s"storage level for RDD; ${defaultParams.storage}")
        .action((x, c) => c.copy(storage=x))
      opt[String]("initMode")
        .text(s"initialization mode (${InitializationMode.values.mkString(",")}), " +
        s"default: ${defaultParams.initializationMode}")
        .action((x, c) => c.copy(initializationMode = InitializationMode.withName(x)))
      arg[String]("<input>")
        .text("input paths to examples")
        .required()
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"DenseKMeans with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val data = sc.sequenceFile[LongWritable, VectorWritable](params.input)

    val storagelevel= params.storage match {
      case 0 => StorageLevel.OFF_HEAP
      case 1 => StorageLevel.DISK_ONLY
      case 2 => StorageLevel.DISK_ONLY_2
      case 3 => StorageLevel.MEMORY_ONLY
      case 4 => StorageLevel.MEMORY_ONLY_2
      case 5 => StorageLevel.MEMORY_ONLY_SER
      case 6 => StorageLevel.MEMORY_ONLY_SER_2
      case 7 => StorageLevel.MEMORY_AND_DISK
      case 8 => StorageLevel.MEMORY_AND_DISK_2
      case 9 => StorageLevel.MEMORY_AND_DISK_SER
      case 10 => StorageLevel.MEMORY_AND_DISK_SER_2
      case _ => StorageLevel.MEMORY_AND_DISK
    }

    val examples = data.map { case (k, v) =>
      var vector: Array[Double] = new Array[Double](v.get().size)
      for (i <- 0 until v.get().size) vector(i) = v.get().get(i)
      Vectors.dense(vector)
    }.persist(storagelevel)

//    val examples = sc.textFile(params.input).map { line =>
//      Vectors.dense(line.split(' ').map(_.toDouble))
//    }.cache()

    val numExamples = examples.count()

    println(s"numExamples = $numExamples.")

    val initMode = params.initializationMode match {
      case Random => KMeans.RANDOM
      case Parallel => KMeans.K_MEANS_PARALLEL
    }

    val model = new KMeans()
      .setInitializationMode(initMode)
      .setK(params.k)
      .setMaxIterations(params.numIterations)
      .run(examples)

    val cost = model.computeCost(examples)

    println(s"Total cost = $cost.")

    sc.stop()
  }
}

