#!/usr/bin/env python2
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mapping from properties to environment variable names
"""
HiBenchEnvPropMappingMandatory=dict(
    JAVA_BIN="java.bin",
    HADOOP_HOME="hibench.hadoop.home",
    HDFS_MASTER="hibench.hdfs.master",
    HADOOP_RELEASE="hibench.hadoop.release",        
    HADOOP_EXAMPLES_JAR="hibench.hadoop.examples.jar", 
    HADOOP_EXECUTABLE="hibench.hadoop.executable", 
    HADOOP_CONF_DIR="hibench.hadoop.configure.dir",
    HIBENCH_HOME="hibench.home",
    HIBENCH_CONF="hibench.configure.dir", 

    REPORT_COLUMN_FORMATS="hibench.report.formats",
    SPARKBENCH_JAR="hibench.sparkbench.jar",
    NUM_MAPS="hibench.default.map.parallelism",
    NUM_REDS="hibench.default.shuffle.parallelism",
    INPUT_HDFS="hibench.workload.input",
    OUTPUT_HDFS="hibench.workload.output",

    REDUCER_CONFIG_NAME="hibench.hadoop.reducer.name",
    MAP_CONFIG_NAME="hibench.hadoop.mapper.name",

    MASTERS="hibench.masters.hostnames",
    SLAVES="hibench.slaves.hostnames",
    )

HiBenchEnvPropMapping=dict(
    SPARK_HOME="hibench.spark.home",
    SPARK_MASTER="hibench.spark.master",
    SPARK_EXAMPLES_JAR="hibench.spark.examples.jar",

    HIVE_HOME="hibench.hive.home",
    HIVE_RELEASE="hibench.hive.release",
    HIVEBENCH_TEMPLATE="hibench.hivebench.template.dir",
    MAHOUT_HOME="hibench.mahout.home",
    MAHOUT_RELEASE="hibench.mahout.release",
    NUTCH_HOME="hibench.nutch.home",
    NUTCH_BASE_HDFS="hibench.nutch.base.hdfs",
    NUTCH_INPUT="hibench.nutch.dir.name.input",
    NUTCH_DIR="hibench.nutch.nutchindexing.dir",
    HIBENCH_REPORT="hibench.report.dir", # set in default
    HIBENCH_REPORT_NAME="hibench.report.name", # set in default
    YARN_NUM_EXECUTORS="hibench.yarn.executor.num",
    YARN_EXECUTOR_CORES="hibench.yarn.executor.cores",
    SPARK_YARN_EXECUTOR_MEMORY="spark.executor.memory",
    SPARK_YARN_DRIVER_MEMORY="spark.driver.memory",
    DATA_HDFS="hibench.hdfs.data.dir",
    # For Sleep workload
    MAP_SLEEP_TIME="hibench.sleep.mapper.seconds",
    RED_SLEEP_TIME="hibench.sleep.reducer.seconds",
    HADOOP_SLEEP_JAR="hibench.sleep.job.jar",
    # For Sort, Terasort, Wordcount
    DATASIZE="hibench.workload.datasize",

    # For hive related workload, data scale
    PAGES="hibench.workload.pages",
    USERVISITS="hibench.workload.uservisits",
    HIVE_INPUT="hibench.workload.dir.name.input",
    HIVE_BASE_HDFS="hibench.hive.base.hdfs",
    # For bayes
    CLASSES="hibench.workload.classes",
    BAYES_INPUT="hibench.bayes.dir.name.input",
    DATATOOLS="hibench.hibench.datatool.dir",
    BAYES_BASE_HDFS="hibench.bayes.base.hdfs",
    NGRAMS="hibench.bayes.ngrams",
    # For kmeans
    INPUT_SAMPLE="hibench.kmeans.input.sample",
    INPUT_CLUSTER="hibench.kmeans.input.cluster",
    NUM_OF_CLUSTERS="hibench.kmeans.num_of_clusters",
    NUM_OF_SAMPLES="hibench.kmeans.num_of_samples",
    SAMPLES_PER_INPUTFILE="hibench.kmeans.samples_per_inputfile",
    DIMENSIONS="hibench.kmeans.dimensions",
    MAX_ITERATION="hibench.kmeans.max_iteration",
    K="hibench.kmeans.k",
    KMEANS_STORAGE_LEVEL="hibench.kmeans.storage_level",
    # For Pagerank
    PAGERANK_BASE_HDFS="hibench.pagerank.base.hdfs",
    PAGERANK_INPUT="hibench.pagerank.dir.name.input",
    BLOCK="hibench.pagerank.block",
    NUM_ITERATIONS="hibench.pagerank.num_iterations",
    PEGASUS_JAR="hibench.pagerank.pegasus.dir",
    PAGERANK_STORAGE_LEVEL="hibench.pagerank.storage_level",
    # For DFSIOE
    RD_NUM_OF_FILES="hibench.dfsioe.read.number_of_files",
    RD_FILE_SIZE="hibench.dfsioe.read.file_size",
    WT_NUM_OF_FILES="hibench.dfsioe.write.number_of_files",
    WT_FILE_SIZE="hibench.dfsioe.write.file_size",
    MAP_JAVA_OPTS="hibench.dfsioe.map.java_opts",
    RED_JAVA_OPTS="hibench.dfsioe.red.java_opts",
    # For NWeight
    MODEL_INPUT="hibench.nweight.model_path",
    EDGES="hibench.workload.edges",
    DEGREE="hibench.nweight.degree",
    MAX_OUT_EDGES="hibench.nweight.max_out_edges",
    NUM_PARTITION="hibench.nweight.partitions",
    STORAGE_LEVEL="hibench.nweight.storage_level",
    DISABLE_KRYO="hibench.nweight.disable_kryo",
    MODEL="hibench.nweight.model",

    # For streaming bench
    STREAMING_TESTCASE="hibench.streambench.testCase",
    COMMON_JAR="hibench.common.jar",

    # prepare
    STREAMING_TOPIC_NAME="hibench.streambench.kafka.topic",
    STREAMING_KAFKA_HOME="hibench.streambench.kafka.home",
    STREAMING_ZKADDR="hibench.streambench.zkHost",
    STREAMING_CONSUMER_GROUP="hibench.streambench.kafka.consumerGroup",
    STREAMING_DATA_DIR="hibench.streambench.datagen.dir",
    STREAMING_DATA1_NAME="hibench.streambench.datagen.data1.name",
    STREAMING_DATA1_DIR="hibench.streambench.datagen.data1.dir",
    STREAMING_DATA1_LENGTH="hibench.streambench.datagen.recordLength",
    STREAMING_DATA2_SAMPLE_DIR="hibench.streambench.datagen.data2_samples.dir",
    STREAMING_DATA2_CLUSTER_DIR="hibench.streambench.datagen.data2_cluster.dir",
    STREAMING_PARTITIONS="hibench.streambench.kafka.topicPartitions",
    DATA_GEN_JAR="hibench.streambench.datagen.jar",

    # metrics reader
    METRICE_READER_SAMPLE_NUM="hibench.streambench.metricsReader.sampleNum",
    METRICS_READER_THREAD_NUM="hibench.streambench.metricsReader.threadNum",
    METRICS_READER_OUTPUT_DIR="hibench.streambench.metricsReader.outputDir",

    # sparkstreaming
    STREAMBENCH_SPARK_JAR="hibench.streambench.sparkbench.jar",
    STREAMBENCH_STORM_JAR="hibench.streambench.stormbench.jar",

    # gearpump
    GEARPUMP_HOME="hibench.streambench.gearpump.home",
    STREAMBENCH_GEARPUMP_JAR="hibench.streambench.gearpump.jar",
    STREAMBENCH_GEARPUMP_EXECUTORS="hibench.streambench.gearpump.executors",

    # flinkstreaming
    HIBENCH_FLINK_MASTER="hibench.flink.master",
    FLINK_HOME="hibench.streambench.flink.home",
    STREAMBENCH_FLINK_JAR="hibench.streambench.flinkbench.jar",
    STREAMBENCH_FLINK_PARALLELISM="hibench.streambench.flink.parallelism",
    
    )

HiBenchPropEnvMapping=dict([(v,k) for k, v in HiBenchEnvPropMapping.items()])
HiBenchPropEnvMappingMandatory=dict([(v,k) for k, v in HiBenchEnvPropMappingMandatory.items()])