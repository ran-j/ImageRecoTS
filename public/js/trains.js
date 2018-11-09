var training = new Array();
var imgs = [];
var classes = [];
var ImgRecoList = [];

const meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);

async function DoTheThing() {
  BuildData().then(() => {
    TrainData();
  })  
}

// (async function(){   
//   setTimeout(function(){
//     PrepareData()
//   },1000)
   
// })();

$( document ).ready(function() {
  PrepareData()
});

$( "#train-button" ).click(async function(){
  $( "#pgBar" ).width('0%')
  $( "#pgBar" ).text(' ');

  $( "#train-button" ).removeClass( "btn btn-primary float-right" ).addClass("btn btn-dark float-right");
  $("#train-button").attr("disabled", true);

  setTimeout(function(){
    BuildData().then(() => {
      TrainData();
    })
  },500)  
  
});

async function PrepareData() {
  imgList.forEach(async (img, index) => {
    console.log(img.ClassName);

    if (!ContainsinArray(classes, img.ClassName)) {
      classes.push(img.ClassName);
    }

    // document.getElementById("imgH").setAttribute("src", img.Path);

    var image = document.createElement("img");

    image.width = '100'
    image.height="100"


    await image.addEventListener(
      "load",
      function() {
        let imgPixel = tf
          .fromPixels(image)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .sub(meanImageNetRGB)
          .reverse(2)
          .expandDims()
          .dataSync();

        console.log(imgPixel);
        imgs.push(imgPixel);

        ImgRecoList.push([img.ClassName, imgPixel]);
      },
      false
    );

    image.src = img.Path;
  });

  console.log(imgs.length, "imgs");
  console.log(classes.length, "classes");
}

async function BuildData() {
  ImgRecoList.forEach((data, i) => {
    var bag = [];

    imgs.forEach((img, ii) => {
      if (FloatCompare(data[1], img)) {
        bag.push(1);
      } else {
        bag.push(0);
      }
    });

    var output_row = new Array(classes.length + 1)
      .join("0")
      .split("")
      .map(parseFloat);
    output_row[classes.findIndex(x => x == data[0])] = 1;
    training.push([bag, output_row]);
  });
}

async function TrainData() {
  // training = shuffle(training);

  console.log(training);

  var train_x = pick(training, 0);
  var train_y = pick(training, 1);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 256,
      activation: "relu",
      inputShape: [train_x[0].length]
    })
  );
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(
    tf.layers.dense({ units: train_y[0].length, activation: "softmax" })
  );
  model.compile({
    optimizer: "rmsprop",
    // optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  const xs = tf.tensor(train_x);
  const ys = tf.tensor(train_y);

  //train model
  await model
    .fit(xs, ys, {
      epochs: 1000,
      batchSize: 100,
      shuffle: true,
      // verbose: 1,
      callbacks: {
        onEpochEnd: async (epoch, log) => {
          console.log(`Epoch ${epoch}: loss = ${log.loss}`);
          EnBar(epoch)
        }
      }
    })
    .then(async () => {
      console.log("Saving model....");
      //Print a text summary of the model's layers.
      model.summary();
      await model.save("http://localhost:8080/model").then(() => {
        console.log(" ");
        console.log("Model Saved.");
        console.log(" ");
        //release memory
        xs.dispose();
        ys.dispose();
        $( "#train-button" ).removeClass( "btn btn-dark float-right" ).addClass("btn btn-primary float-right");
        $("#train-button").attr("disabled", false);
        alert('Model trained')
      });
    });
}

function ContainsinArray(A, value) {
  if (!A) {
    return false;
  }
  return A.indexOf(value) > -1;
}

function FloatCompare(a1, a2, compareOrder) {
  if (a1.length !== a2.length) {
    return false;
  }

  return a1.every(function(value, index) {
    if (compareOrder) {
      return value === a2[index];
    } else {
      return a2.indexOf(value) > -1;
    }
  });
}

function pick(matrix, col) {
  let column = [];
  for (let i = 0; i < matrix.length; i++) {
    column.push(matrix[i][col]);
  }
  return column;
}

function shuffle(array) {
  var currentIndex = array.length,
    temporaryValue,
    randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {
    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

function multiDimensionalUnique(arr) {
  var uniques = [];
  var itemsFound = {};
  for (var i = 0, l = arr.length; i < l; i++) {
    var stringified = JSON.stringify(arr[i]);
    if (itemsFound[stringified]) {
      continue;
    }
    uniques.push(arr[i]);
    itemsFound[stringified] = true;
  }
  return uniques;
}

function EnBar(val){
  let porcent = R3(999,100,parseInt(val))

  $( "#pgBar" ).text(porcent+'%');
  $( "#pgBar" ).width(porcent+'%')  
}

function R3(A,B,X){
  return Math.round((B * X) / A)
}