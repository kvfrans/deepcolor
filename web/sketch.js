var linecanvas = document.getElementById("line");
var linectx = linecanvas.getContext("2d");
linectx.lineCap = "round";
linectx.lineJoin = "round";
linectx.lineWidth = 3;

var colorcanvas = document.getElementById("color");
var colorctx = colorcanvas.getContext("2d");
colorctx.lineCap = "round";
colorctx.lineJoin = "round";
colorctx.lineWidth = 6;


colorctx.beginPath();
colorctx.rect(0, 0, 512, 512);
colorctx.fillStyle = "white";
colorctx.fill();

var lastX;
var lastY;

var mouseX;
var mouseY;
var canvasOffset = $("#color").offset();
var offsetX = canvasOffset.left;
var offsetY = canvasOffset.top;
var isMouseDown = false;


function handleMouseDown(e) {
    mouseX = parseInt(e.clientX - offsetX);
    mouseY = parseInt(e.clientY - offsetY);

    // Put your mousedown stuff here
    lastX = mouseX;
    lastY = mouseY;
    isMouseDown = true;
}

function handleMouseUp(e) {
    mouseX = parseInt(e.clientX - offsetX);
    mouseY = parseInt(e.clientY - offsetY);

    // Put your mouseup stuff here
    isMouseDown = false;
}
function handleMouseOut(e) {
    mouseX = parseInt(e.clientX - offsetX);
    mouseY = parseInt(e.clientY - offsetY);

    // Put your mouseOut stuff here
    isMouseDown = false;
}

function handleMouseMove(e)
{
    mouseX = parseInt(e.clientX - offsetX);
    mouseY = parseInt(e.clientY - offsetY);
    // Put your mousemove stuff here
    if(isMouseDown)
    {
        if(mode == "pen")
        {
            linectx.beginPath();
            linectx.globalCompositeOperation = "source-over";
            linectx.moveTo(lastX, lastY);
            linectx.lineTo(mouseX, mouseY);
            linectx.stroke();
        }
        else if(mode == "eraser")
        {
            linectx.beginPath();
            linectx.globalCompositeOperation = "destination-out";
            linectx.arc(lastX, lastY, 10, 0, Math.PI * 2, false);
            linectx.fill();
        }
        else
        {
            colorctx.beginPath();
            colorctx.strokeStyle = mode;
            colorctx.globalCompositeOperation = "source-over";
            colorctx.moveTo(lastX, lastY);
            colorctx.lineTo(mouseX, mouseY);
            colorctx.stroke();
        }
        lastX = mouseX;
        lastY = mouseY;
    }
}

$("#line").mousedown(function (e) {
    handleMouseDown(e);
});
$("#line").mousemove(function (e) {
    handleMouseMove(e);
});
$("#line").mouseup(function (e) {
    handleMouseUp(e);
});
$("#line").mouseout(function (e) {
    handleMouseOut(e);
});

var mode = "pen";
$("#pen").click(function () {
    mode = "pen";
});
$("#eraser").click(function () {
    mode = "eraser";
});

$("#submit").click(function () {

    // change non-opaque pixels to white
    var imgData = linectx.getImageData(0,0,512,512);
    var data = imgData.data;
    var databackup = data.slice(0);
    for(var i = 0; i < data.length; i+=4)
    {
        if(data[i+3]<255)
        {
            data[i]=255;
            data[i+1]=255;
            data[i+2]=255;
            data[i+3]=255;
        }
    }

    linectx.putImageData(imgData,0,0);

    var dataURL = linecanvas.toDataURL("image/jpg");
    var dataURLc = colorcanvas.toDataURL("image/jpg");

    imgData = linectx.getImageData(0,0,512,512);
    data = imgData.data;
    for(var i = 0; i < data.length; i++)
    {
        data[i] = databackup[i];
    }
    linectx.putImageData(imgData,0,0);
    // console.log(dataURL)

    $.ajax({
        url: '/upload_canvas',
        type: "POST",
        data: {colors: dataURLc, lines: dataURL},
        // data: {lines: "meme"},
        success: function (result) {
            console.log("Upload complete!");
        },
        error: function (error) {
            console.log("Something went wrong!");
        }
    });
});

















/* config size & modify */
var config = {
    size: 250
};

(function(){
    document.getElementById('colorwheel').style.height = config.size+"px";
    document.getElementById('colorwheel').style.width = config.size+"px";
    document.getElementById('gamma_slider').style.height = (config.size-20)+"px";
})();

/* render color wheel canvas#surface */
(function () {
    var el = document.getElementById('surface'),
        context = el.getContext('2d'),
        width = el.parentNode.offsetWidth,
        height = el.parentNode.offsetHeight,
        cx = width / 2,
        cy = height / 2,
        radius = width / 2.3,
        imageData,
        pixels,
        hue, sat, value,
        i = 0,
        x, y, rx, ry, d,
        f, g, p, u, v, w, rgb;

    el.width = width;
    el.height = height;
    imageData = context.createImageData(width, height);
    pixels = imageData.data;

    for (y = 0; y < height; y = y + 1) {
        for (x = 0; x < width; x = x + 1, i = i + 4) {
            rx = x - cx;
            ry = y - cy;
            d = rx * rx + ry * ry;
            if (d < radius * radius) {
                hue = 6 * (Math.atan2(ry, rx) + Math.PI) / (2 * Math.PI);
                sat = Math.sqrt(d) / radius;
                g = Math.floor(hue);
                f = hue - g;
                u = 255 * (1 - sat);
                v = 255 * (1 - sat * f);
                w = 255 * (1 - sat * (1 - f));
                pixels[i] = [255, v, u, u, w, 255, 255][g];
                pixels[i + 1] = [w, 255, 255, v, u, u, w][g];
                pixels[i + 2] = [u, u, w, 255, 255, v, u][g];
                pixels[i + 3] = 255;
            } else {
                pixels[i] = 255;
                pixels[i + 1] = 255;
                pixels[i + 2] = 255;
                pixels[i + 3] = 255;
            }
        }
    }

    context.putImageData(imageData, 0, 0);
})();

/* some globals */
var color_selection;
var rgb = [255, 255, 255, 255];
var gamma = 1;

/* bind click to #surface to set rgb and color_selection
 * bind click to #gamma_slider to set gamma value for color
 * finally display changes in #colorblock
 */
(function () {
    var el = document.getElementById('surface');

    el.addEventListener('click', function (e) {

        var x = e.pageX - el.offsetLeft;
        var y = e.pageY - el.offsetTop;

        data = el.getContext('2d').getImageData(x, y, 1, 1).data;
        console.log(data[0] + ',' + data[1] + ',' + data[2]);

        rgb = data;

        var hex = "#" + ("000000" + rgbToHex(data[0] * gamma, data[1] * gamma, data[2] * gamma)).slice(-6);
        color_selection = hex;
        document.getElementById('colorblock').style.backgroundColor = color_selection;
        document.getElementById('color_selection').innerHTML = color_selection;
        mode = color_selection
    }, true);

    var gamma_slider = document.getElementById('gamma_wrapper');

    gamma_slider.addEventListener('click',function(e){
        var y = e.pageY - gamma_slider.offsetTop;
        var gamma_val = y / parseFloat(gamma_slider.clientHeight);

        //alert(1-gamma_val);

        gamma_val = parseFloat(1.15-gamma_val);

        if(gamma_val > 1){
            gamma = 1;
        }else{
            gamma = gamma_val;
        }

        var hex = "#" + ("000000" + rgbToHex(rgb[0] * gamma, rgb[1] * gamma, rgb[2] * gamma)).slice(-6);

        color_selection = hex;

        document.getElementById('colorblock').style.backgroundColor = color_selection;
        document.getElementById('color_selection').innerHTML = color_selection;
        mode = color_selection
    });

})();

/* Utilities */
function rgbToHex(r, g, b) {
    if (r > 255 || g > 255 || b > 255) throw "Invalid color component";
    return ((r << 16) | (g << 8) | b).toString(16);
}
