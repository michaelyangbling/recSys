package project
//import com.amazonaws.auth.{BasicAWSCredentials}
//import com.amazonaws.services.s3.AmazonS3Client
//import com.amazonaws.services.s3.model.GetObjectRequest



import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.io.Source
import scala.util.control.Breaks._
import scala.collection.mutable.Map
import scala.math.log1p
object CF {

//  def updateMap(path: String, songIntMap: Map[String, Int], userIntMap: Map[String, Int], numLines: Int
//               , arr: Array[String]): Unit ={  //numlines: lines to include
//
////    turn song and user IDs in file to songIntMap and userIntMap
////-1 numLines to read all data
//
//    var count=0
//
////    val credentials = new BasicAWSCredentials("myKey", "mySecretKey")
////    val s3Client = new AmazonS3Client(credentials)
////    val s3Object = s3Client.getObject(new GetObjectRequest("my-bucket", "input.txt"))
////    val myData = Source.fromInputStream(s3Object.getObjectContent())
//
////    val runid = myData.getLines().mkString
//
////    val file= Source.fromFile("s3:///ds5230-sparkyang/bigTrain.txt")
//    try {
//      println(99)
//      for (line <- arr) {
//        if (count == numLines) {
//          //reach number of lines to include
//          break
//        }
//        if (line.length!=0) { //empty line due to concatenating files
//
//
//          var contents = line.split('\t')
//          var user = contents(0) //original String ID
//          var song = contents(1)
//
//          if (!userIntMap.contains(user)) { //if not in map, put key->value pair to map
//            userIntMap(user) = userIntMap.size
//          }
//
//          if (!songIntMap.contains(song)) {
//            songIntMap(song) = songIntMap.size
//          }
//
//          count = count + 1
//        }
//      }
//    }
//
//  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val conf = new SparkConf().setAppName("Word Count")
    val sc = new SparkContext(conf)
//    val songIntMap: Map[String, Int] = scala.collection.mutable.Map[String, Int]()
//    val userIntMap: Map[String, Int] = scala.collection.mutable.Map[String, Int]()

		// Delete output directory, only to ease local development; will not work on AWS. ===========
//    val hadoopConf = new org.apache.hadoop.conf.Configuration
//    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
//    try { hdfs.delete(new org.apache.hadoop.fs.Path(args(1)), true) } catch { case _: Throwable => {} }
		// ================

    // Load and parse the data
    println(args(0))
    println(args(1))
    val train = sc.textFile(args(0))
    val test = sc.textFile(args(1))
    //better to persist train.map
    val userIntMap = train.map(_.split('\t') match { case Array(user, song, rate) =>
      ( user, true )
    }).reduceByKey((a, b)=>true).zipWithIndex().map(pair=>(pair._1._1, pair._2.toInt)).collectAsMap()
    //after zip: ( (uid, true), index )
//    updateMap(args(0), songIntMap, userIntMap, -1, train.collect() )

    val songIntMap = train.map(_.split('\t') match { case Array(user, song, rate) =>
      ( song, true )
    }).reduceByKey((a, b)=>true).zipWithIndex().map(pair=>(pair._1._1, pair._2.toInt)).collectAsMap()
    println(1)


    val userIntMapBroad= sc.broadcast(userIntMap) //broadcast stringId: int map to all worker nodes
    println(2)
    val songIntMapBroad=sc.broadcast(songIntMap)

//    ratings = data.map(lambda l: l.split('\t')).map(
//      lambda l: [ userIntMap.value[ l[0] ] , songIntMap.value[ l[1] ] , l[2] ])\
//      .map(lambda l: Rating(int( l[0]  ), int( l[1] ), float(l[2])) )



    println(3)
    //map String to Int
    val trainRatings = train.map(_.split('\t') match { case Array(user, song, rate) =>
       ( ( userIntMapBroad.value.get(user).get, songIntMapBroad.value.get(song).get ), rate )
    }).map( tup => Rating(tup._1._1, tup._1._2, log1p(tup._2.toDouble )) )
    println(4)
    //filter: only calculate RMSE for met entries
    val testRatings = test.map(_.split('\t') match { case Array(user, song, rate) =>
      ( (user, song), rate )}).filter( tup =>
      userIntMapBroad.value.contains(tup._1._1) &&  songIntMapBroad.value.contains(tup._1._2)  ).map( tup =>
      ( ( userIntMapBroad.value.get( tup._1._1).get, songIntMapBroad.value.get( tup._1._2).get ), tup._2 )).map( tup =>
      Rating(tup._1._1, tup._1._2, log1p( tup._2.toDouble ) ) )




    // Build the recommendation model using ALS
    val rank = 10
    val numIterations = 10

    val model = ALS.train(trainRatings, rank, numIterations, 0.01)
    println(5)
    // Evaluate the model on rating data
    val usersSongs = trainRatings.map { case Rating(user, song, rate) =>
      (user, song)
    }
    println(6)
//    usersSongs.saveAsTextFile(args(2))

    val predictions =
      model.predict(usersSongs).map { case Rating(user, song, rate) =>
        ((user, song), rate)
      }
//    predictions.saveAsTextFile(args(3))

    val ratesAndPreds = trainRatings.map { case Rating(user, song, rate) =>
      ((user, song), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map { case (_, (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()


    val usersSongs2 = testRatings.map { case Rating(user, song, rate) =>
      (user, song)
    }

    //    usersSongs.saveAsTextFile(args(2))

    val predictions2 =
      model.predict(usersSongs2).map { case Rating(user, song, rate) =>
        ((user, song), rate)
      }
    //    predictions.saveAsTextFile(args(3))

    val ratesAndPreds2 = testRatings.map { case Rating(user, song, rate) =>
      ((user, song), rate)
    }.join(predictions2)

    val testMSE = ratesAndPreds2.map { case (_, (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
    println(7)




    
    println(s"train Mean Squared Error = $MSE")
    println(s"test Mean Squared Error = $testMSE")

    // Save and load model
    //model.save(sc, "target/tmp/myCollaborativeFilter")
    //val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
  }
}

//train Mean Squared Error = 0.027977990345466074
//test Mean Squared Error = 0.6828938596336082



//train Mean Squared Error = 0.15145571607849187
//test Mean Squared Error = 0.38940156327679415