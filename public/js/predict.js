//Triggering change event after the Image is selected from the Browse Button 
$("#image-selector").change(function(){
    try {
        let reader = new FileReader();
    
        reader.onload = function(){
            let dataURL = reader.result;
            $("#selected-image").attr("src",dataURL);
            $("#prediction-list").empty();
            $('#selected-image').show();
        }
        
        let file = $("#image-selector").prop('files')[0];
        
        reader.readAsDataURL(file);

    } catch (error) {
        $('#selected-image').hide();
    }    
});

let model;
var imgs = [];
let tensor;

(async function(){
    model = await tf.loadModel('http://localhost:8080/store/model.json');   
    PrepareData() 
})();

//define a 1D tensor with ImageNet mean RGB Values..
let meanImageNetRGB = tf.tensor1d([123.68,116.779,103.939]);

$("#predict-button").click(async function(){
    $("#predict-button").attr("disabled", true);

    $("#prediction-list").hide();
    
    $( "#predict-button" ).removeClass( "btn btn-primary" ).addClass("btn btn-dark");

    setTimeout(async function(){
        let image = $('#selected-image').get(0);
    
        //convert the image object to a tensor by resizing it and Normalizing it using the ImageNet mean RGB values
        tensor = tf.fromPixels(image)
                        .resizeNearestNeighbor([224,224])
                        .toFloat().sub(meanImageNetRGB)
                        .reverse(2)
                        .expandDims()
                        .dataSync();

        
        const bowData = await bow(tensor);

        console.log(bowData)

        var data = tf.tensor2d(bowData, [1, bowData.length]);
                
        let prediction = await model.predict(data).data();
        
        console.log(prediction)

        let top5 = Array.from(prediction)    
        .map(function(p,i){
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function(a,b){
            return b.probability-a.probability;
        }).slice(0,5);

        $("#prediction-list").empty();
        $("#prediction-list").show();

        top5.forEach(function(p,i){

            if(i == 0){
                $("#prediction-list").append(`<li class="list-group-item list-group-item-success">${p.className}:${p.probability.toFixed(6)}</li>`);
            
            }else if (i == 1){
                $("#prediction-list").append(`<li class="list-group-item list-group-item-secondary">${p.className}:${p.probability.toFixed(6)}</li>`);
                
            }else{
                $("#prediction-list").append(`<li class="list-group-item list-group-item-light">${p.className}:${p.probability.toFixed(6)}</li>`);
            }
            
        });

        // $('#selected-image').hide();

        $( "#predict-button" ).removeClass("btn btn-dark").addClass("btn btn-primary");

        $("#predict-button").attr("disabled", false);
    },500)    
});


async function bow(imgatual){
    var bag = new Array(imgs.length + 1).join('0').split('').map(parseFloat);
    
    imgs.forEach((img, i) => {
        if (arraysEqual(imgatual,img)) {
            bag[i] = 1;     
        }
    });
    
    return bag    
}

async function PrepareData() {

    await imgList.forEach(async (img, index) => {
      
      var image = document.createElement("img");

        image.width = '100'
        image.height="100"
        
        image.addEventListener("load",function() {
          
            let imgPixel = tf.fromPixels(image)
                        .resizeNearestNeighbor([224, 224])
                        .toFloat()
                        .sub(meanImageNetRGB)
                        .reverse(2)
                        .expandDims()
                        .dataSync();
            
            console.log(imgPixel);

            imgs.push(imgPixel);  
          
        });  

        image.src = img.Path;
    });
  
    console.log(imgs.length, 'imgs') 
}

function arraysEqual(a1, a2, compareOrder) {
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