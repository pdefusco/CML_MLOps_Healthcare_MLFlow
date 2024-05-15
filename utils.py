#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata


class HealthDataGen:

    '''Class to Generate Biomarkers Data'''

    def __init__(self, username, dbname, storage, connectionName):
        self.username = username
        self.storage = storage
        self.dbname = dbname
        self.connectionName = connectionName

    def biomarkersDataGen(self, spark, shuffle_partitions_requested = 10, partitions_requested = 10, data_rows = 10000):

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("cd8_perc", "float", minValue=0, maxValue=1, random=True)
                    .withColumn("cd19_perc", "float", minValue=0, maxValue=1, random=True)
                    .withColumn("cd45_abs_count", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("cd3_perc", "float", minValue=0, maxValue=1, random=True)
                    .withColumn("cd19_abs_count", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("iga", "float", minValue=1, maxValue=1000, random=True)
                    .withColumn("c3", "float", minValue=1, maxValue=1000, random=True)
                    .withColumn("cd4_abs_count", "float", minValue=1, maxValue=10000, random=True)
                    .withColumn("cd16cd56_perc", "float", minValue=0, maxValue=1, random=True)
                    .withColumn("cd8_abs_count", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("cd4_ratio_cd8", "float", minValue=0, maxValue=1, random=True)
                    .withColumn("age", "int", minValue=1, maxValue=15, random=True)
                    .withColumn("cd3_abs_count", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("igm", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("cd4_perc", "float", minValue=0, maxValue=1, random=True)
                    .withColumn("tige", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("ch50", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("c4", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("cd16cd56_abs_count", "float", minValue=1, maxValue=100000, random=True)
                    .withColumn("allergy_hist", "string", values=["0", "1"], random=True, weights=[7, 3])
                    .withColumn("lung_compl", "string", values=["0", "1"], random=True, weights=[8, 2])
                    .withColumn("gender", "string", values=["0", "1"], random=True, weights=[5, 5])
                    .withColumn("asthmatic_bronchitis", "string", values=["0", "1"], random=True, weights=[7, 3])
                    )

        df = fakerDataspec.build()

        df = df.withColumn("allergy_hist", df["allergy_hist"].cast(IntegerType()))
        df = df.withColumn("lung_compl", df["lung_compl"].cast(IntegerType()))
        df = df.withColumn("gender", df["gender"].cast(IntegerType()))
        df = df.withColumn("asthmatic_bronchitis", df["asthmatic_bronchitis"].cast(IntegerType()))

        return df


    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        """
        Method to save credit card transactions df as csv in cloud storage
        """

        df.write.format("csv").mode('overwrite').save(self.storage + "/health_biomarkers_demo/" + self.username)


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the BIOMARKER RECORDS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.BIOMARKERS_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.BIOMARKERS_{1}".format(self.dbname, self.username))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()
