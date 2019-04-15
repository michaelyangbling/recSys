package project

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.io.Source

object CF {

  def updateMap(path: String, songIntMap: Map[String, Int], userIntMap: Map[String, Int], numLines: Int): Unit ={  //numlines: lines to include

//    turn song and user IDs in file to songIntMap and userIntMap
//-1 numLines to read all data

    var count=0
    val file= Source.fromFile(path)
    try {
      for (line <- file.getLines) {
        if (count == numLines) {
          //reach number of lines to include
          break
        }
        if (contents.length==0){   //empty line due to concatenating files
          continue
        }
        var contents = line.split('\t')
        var user = contents(0) //original String ID
        var song = contents(1)

        if (!userIntMap.contains(user)) { //if not in map, put key->value pair to map
          userIntMap(user) = userIntMap.size()
        }

        if (!songIntMap.contains(song)) {
          songIntMap(song) = songIntMap.size()
        }

        count = count + 1

      }
    }

    finally{
      file.close()
    }

  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val conf = new SparkConf().setAppName("Word Count")
    val sc = new SparkContext(conf)
    val songIntMap: Map[String, Int] = Map()
    val userIntMap: Map[String, Int] = Map()

		// Delete output directory, only to ease local development; will not work on AWS. ===========
//    val hadoopConf = new org.apache.hadoop.conf.Configuration
//    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
//    try { hdfs.delete(new org.apache.hadoop.fs.Path(args(1)), true) } catch { case _: Throwable => {} }
		// ================

    // Load and parse the data
    val train = sc.textFile(args(0))
    val test = sc.textFile(args(1))

    updateMap(args(0), songIntMap, userIntMap, -1)


    val userIntMapBroad= sc.broadcast(userIntMap) //broadcast stringId: int map to all worker nodes
    val songIntMapBroad=sc.broadcast(songIntMap)


//    ratings = data.map(lambda l: l.split('\t')).map(
//      lambda l: [ userIntMap.value[ l[0] ] , songIntMap.value[ l[1] ] , l[2] ])\
//      .map(lambda l: Rating(int( l[0]  ), int( l[1] ), float(l[2])) )





    val trainRatings = train.map(_.split('\t') match { case Array(user, song, rate) =>
       ( ( userIntMapBroad.value.get(user), songIntMapBroad.value.get(song)), rate ) //map String to Int
    }).map( tup => Rating(tup._1._1, tup._1._2, tup._2.toDouble ) )


    val testRatings = test.map(_.split('\t') match { case Array(user, song, rate) =>
      ( (user, song), rate )  //filter: only calculate RMSE for met entries
    }).filter( tup => userIntMapBroad.value.contains(tup._1._1) &&  songIntMapBroad.value.contains(tup._1._2)  )
    ).map( tup => ( ( userIntMapBroad.value.get( tup._1._1 ), songIntMapBroad.value.get( tup._1._2 ) ), tup._2 )
    ).map( tup => Rating(tup._1._1, tup._1._2, tup._2.toDouble ) )




    // Build the recommendation model using ALS
    val rank = 10
    val numIterations = 10

    val model = ALS.train(trainRatings, rank, numIterations, 0.01)

    // Evaluate the model on rating data
    val usersSongs = trainRatings.map { case Rating(user, song, rate) =>
      (user, song)
    }

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




    
    println(s"train Mean Squared Error = $MSE")
    println(s"test Mean Squared Error = $testMSE")

    // Save and load model
    //model.save(sc, "target/tmp/myCollaborativeFilter")
    //val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
  }
}