import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.IntegerType



object Main {

  private val replace_null_with_unknown = udf((x: String) => {
    var res = new String
    if (x == null || x == "Unknow") res = "unknown"
    else res = x
    res
  })

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL 2008 dataset")
      .config("spark.master", "local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    var df = spark.read.option("header",value = true).csv("src/main/resources/2008.csv")

    // We separate our target_variable from the rest of the dataset, saving it in a different one
    println()
    println("--------------------------------- We separate our target variable from the rest of the dataset, saving it in a different one -----------------------------------------------")
    var t_col = df.select("ArrDelay")

    // We delete the columns we do not need
    println("--------------------------------- We delete the columns we do not need -----------------------------------------------")
    val columns_to_drop = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "ArrDelay")
    df = df.drop(columns_to_drop:_*)

    // We delete all the rows that contain cancelled flights, since this will not be useful for our prediction goal
    println("--------------------------------- We delete all the rows that contain cancelled flights, since this will not be useful for our prediction goal -----------------------------------------------")
    df = df.filter("Cancelled == 0")

    // Therefore we eliminate the "CancellationCode" and "Cancelled" columns
    println("--------------------------------- Therefore we eliminate the \"CancellationCode\" and \"Cancelled\" columns -----------------------------------------------")
    df = df.drop("Cancelled","CancellationCode")

    // We check for null values in the "TailNum" column and swap them with "unknown"
    println("--------------------------------- We check for null values in the \"TailNum\" column and swap them with \"unknown\" -----------------------------------------------")
    spark.udf.register("replace_null_with_unknown", replace_null_with_unknown)
    df = df.withColumn("tailNum", replace_null_with_unknown(col("tailNum")))

    // We check for missing numerical values in the each column of the dataset
    println("--------------------------------- We check for missing numerical values in the each column of the dataset -----------------------------------------------")
    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      println(column)
      df.filter(col(column).contains("NA")).show()
    }

    // Numerical columns for mean imputer and most frequent imputer
    println("--------------------------------- Numerical columns to be transformed: -----------------------------------------------")
    val num_cols_mean = Array("DepTime","CRSDepTime","CRSArrTime","CRSElapsedTime","DepDelay","Distance","TaxiOut")
    val num_cols_mf = Array("Year","Month","DayofMonth","DayOfWeek")

    // We cast to Integer every numerical column in order to be able to use the imputer
    println("--------------------------------- We cast to Integer every numerical column in order to be able to use the imputer -----------------------------------------------")
    for (i <- 0 until  df.columns.length){
      val colName = df.columns(i)
      if (num_cols_mean.contains(colName) || num_cols_mf.contains(colName))
        df = df.withColumn(colName,col(colName).cast(IntegerType))
    }

    // We apply the most frequent imputer for the columns Year Month DayOfMonth and DayOfWeek
    println("--------------------------------- We apply the most frequent imputer for the columns \"Year\",\"Month\",\"DayofMonth\",\"DayOfWeek\" -----------------------------------------------")
    val imputer = new Imputer()
      .setInputCols(num_cols_mf)
      .setOutputCols(num_cols_mf)
      .setStrategy("mode")
    df = imputer.fit(df).transform(df)

    // We apply the mean imputer for the rest of the numerical columns
    println("--------------------------------- We apply the mean imputer for the columns \"DepTime\",\"CRSDepTime\",\"CRSArrTime\",\"CRSElapsedTime\",\"DepDelay\",\"Distance\",\"TaxiOut\" -----------------------------------------------")
    imputer.setInputCols(num_cols_mean).setOutputCols(num_cols_mean).setStrategy("mean")
    df = imputer.fit(df).transform(df)

    df.show

    // Declaration of the indexer that will transform entries to integer values
    println("--------------------------------- Declaration of the indexer that will transform entries to integer values -----------------------------------------------")
    val indexer = new StringIndexer()

    // Declaration of the one hot encoder that will process the categorical variables
    println("--------------------------------- Declaration of the one hot encoder that will process the categorical variables -----------------------------------------------")
    val ohe = new OneHotEncoder()


    df = indexer.fit(df).transform(df)
    df = ohe.fit(df).transform(df)

    df.show


  }

}

//    // Change the name of the columns for the sake of the code comprehension
//    for (i <- 0 until df.columns.length) {
//      var colName = df.columns(i)
//      var newColName = df.select(colName).first.getString(0)
//      df = df.withColumnRenamed(colName, newColName)
//    }

//    // We delete the first row, as we do not need it anymore, given the fact that we already renamed each column
//    val first_row = df.first()
//    df = df.filter(row => row != first_row)
