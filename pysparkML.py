!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://www-eu.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz
!tar xf spark-2.4.4-bin-hadoop2.7.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"

import findspark

findspark.init("spark-2.4.4-bin-hadoop2.7")# SPARK_HOME

import pyspark
from pyspark import SparkContext
sc =SparkContext()

from pyspark.sql import Row
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)
sqlContext.setConf('spark.sql.shuffle.partitions', '7')

from pyspark.sql import SparkSession
import pyspark.sql as sparksql
from pyspark import SparkFiles

spark = SparkSession.builder.appName('stroke').getOrCreate()

from google.colab import drive
drive.mount('/content/gdrive')
root_path = 'gdrive/My Drive/'


train = sqlContext.read.csv(SparkFiles.get("/content/"+root_path+"train_2v.csv"), header=True, inferSchema=True)



# explain about Infer Schema
train.groupBy('label').count().show()

train.printSchema()

train.dtypes


train.describe().show()

# create DataFrame as a temporary view for SQL queries
train.createOrReplaceTempView('table')

# import all from 'sql.types'
from pyspark.sql.types import *

#sql query to find the number of people in specific work_type who have had stroke and not

spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE label == 1 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show()

spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE label == 0 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show()

# People having stroke are mostly in Private sector or self-employed



#sql query to find the number of people in each gender who have had stroke and not

spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Male') as percentage FROM table WHERE label== 1 AND gender = 'Male' GROUP BY gender").show()


spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Female') as percentage FROM table WHERE label= 1 AND gender = 'Female' GROUP BY gender").show()


# check % stroke had occured for people who are more than 50 years old

spark.sql("SELECT COUNT(age)*100/(SELECT COUNT(age) FROM table WHERE label ==1) as percentage FROM table WHERE label == 1 AND age>=50").show()

train.describe().show()

train.describe('Residence_type').show()

# fill in missing values for smoking status
# As this is categorical data, we will add one data type "No Info" for the missing one
train_f = train.na.fill('No Info', subset=['smoking_status'])

train_f.groupBy('label').count().show()

# fill in miss values for bmi 
# as this is numecial data , we will simple fill the missing values with mean

from pyspark.sql.functions import mean

mean = train_f.select(mean(train_f['bmi'])).collect()
mean_bmi = mean[0][0]
train_f = train_f.na.fill(mean_bmi,['bmi'])

train_f.describe().show()


#To wrap all of that Spark ML represents such a workflow as a Pipeline, 
# which consists of a sequence of PipelineStages to be run in a specific order.

# StringIndexer -> OneHotEncoder -> VectorAssembler

# indexing all categorical columns in the dataset
from pyspark.ml.feature import StringIndexer
indexer1 = StringIndexer(inputCol="gender", outputCol="genderIndex")
indexer2 = StringIndexer(inputCol="ever_married", outputCol="ever_marriedIndex")
indexer3 = StringIndexer(inputCol="work_type", outputCol="work_typeIndex")
indexer4 = StringIndexer(inputCol="Residence_type", outputCol="Residence_typeIndex")
indexer5 = StringIndexer(inputCol="smoking_status", outputCol="smoking_statusIndex")


# Doing one hot encoding of indexed data
from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["genderIndex","ever_marriedIndex","work_typeIndex","Residence_typeIndex","smoking_statusIndex"],
                                 outputCols=["genderVec","ever_marriedVec","work_typeVec","Residence_typeVec","smoking_statusVec"])


# combines a given list of columns into a single vector column to train ML model

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['genderVec',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedVec',
 'work_typeVec',
 'Residence_typeVec',
 'avg_glucose_level',
 'bmi',
 'smoking_statusVec'],outputCol='features')


# create a pipeline
from pyspark.ml import Pipeline

pipeline  = Pipeline(stages = [indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler] )
pipelineModel = pipeline.fit(train_f)
model = pipelineModel.transform(train_f)
# model.take(1)

from pyspark.ml.linalg import DenseVector

input_data = model.rdd.map(lambda x: (x["label"], DenseVector(x["features"])))


#create train dtaa as a DataFrame

df_train = sqlContext.createDataFrame(input_data, ["label", "features"])
df_train.show(2)

# splitting training and validation data
train_data,test_data = df_train.randomSplit([0.8,0.2], seed=1234)



### LOGISTIC REGRESSION

from pyspark.ml.classification import LogisticRegression

# Initialize 'lr'

lr = LogisticRegression(labelCol = 'label',featuresCol='features', maxIter=10, regParam=0.3)


# Calclate time to train the data
from time import *
start_time = time()

# Fit the data to the model
linearModel = lr.fit(train_data)

end_time = time()

elapsed_time = end_time - start_time

print("time to train model: %.3f seconds" % elapsed_time)


# print coefficients and interception for the regression

print("Coefficients:" + str(linearModel.coefficients))
print("Intercept:" + str(linearModel.intercept))


predictions = linearModel.transform(test_data)

selected = predictions.select("label","prediction","probability")
selected.show(20)

#evaluate the model

cm = predictions.select("label",'prediction')
cm.groupby("label").agg({"label":"count"}).show()

cm.groupby("prediction").agg({"prediction":"count"}).show()

cm.filter(cm.label == cm.prediction).count() /cm.count()

def accuracy_m(model):
    predictions = model.transform(test_data)
    cm = predictions.select("label", "prediction")
    acc = cm.filter(cm.label ==cm.prediction).count() /cm.count()
    print("Model accuracy: %.3f%%" % (acc*100))
    
accuracy_m(model = linearModel)

# use ROC

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model

evaluator = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")
print(evaluator.evaluate(predictions))
print(evaluator.getMetricName())

# tune hyperparameter

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.01,0.5]).build())


# Calculate time taken to train the data
from time import *
start_time = time()

# create 5 fold crossValidator

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run Cross Validations

cvModel = cv.fit(train_data)

end_time = time()
elapsed_time = end_time - start_time

print("time to train model: %.3f seconds" % elapsed_time)

accuracy_m(model = cvModel)

bestModel = cvModel.bestModel
bestModel.extractParamMap()

#### DECISION TREE

from pyspark.ml.classification import DecisionTreeClassifier
dtc = DecisionTreeClassifier(labelCol='label',featuresCol='features')

from pyspark.ml import Pipeline
# pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, dtc])


### Calculate time to train the data
from time import *
start_time = time()

# training model pipeline with data
DecisionT = dtc.fit(train_data)

end_time = time()

elapsed_time = end_time - start_time

print("time to train model: %.3f seconds" % elapsed_time)


# making prediction on model with validation data
dtc_predictions = DecisionT.transform(test_data)

# Select example rows to display.
dtc_predictions.select("label","prediction","probability").show(20)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Select (prediction, true label) and compute test error

acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dtc_acc = acc_evaluator.evaluate(dtc_predictions)

print('A Decision Tree algorithm had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))

#create train dataa as a DataFrame

df_test = sqlContext.createDataFrame(input_data,['feature'])
df_test.show(2)
