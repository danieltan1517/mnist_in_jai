main :: () {
  if !load_neural_network("mnist.nn") {
    print("Error.\n");
  }

  mnist_image();
}

mnist_image :: () {
  x: s32;
  y: s32;
  comp: s32;
  data := stbi_load("MNIST/1_3.jpg", *x, *y, *comp, 0);
  buffer: [] u8;
  buffer.data = data;
  buffer.count = x * y;
  image: [784] float;

  for i: 0..783 {
    image[i] = clamp((xx buffer[i]) / 256.0, 0.0, 1.0);
  }

  t := get_time();
  results := forward_propagate_cpu(image);
  t = get_time() - t;
  print("%. time taken: %\n", results, t);

  t = get_time();
  results = forward_propagate_sse(image);
  t = get_time() - t;
  print("%. time taken: %\n", results, t);
}

load_neural_network :: (file_name: string) -> bool {
  bytes, success := read_entire_file(file_name);
  if !success then {
    print("Error. Unable to open file.\n");
    return false;
  }

  if !begins_with(bytes, "MNIST_MODEL") {
    print("Incorrect File Format.\n");
    return false;
  }

  bytes = advance(bytes, 11);
  float_data: *float = cast(*float)bytes.data;

  memcpy(*mnist_model, bytes.data, size_of(MNIST_Model));
  return true;
}

compute_flag :: enum {
  cpu;
  sse;
  avx2;
}

forward_propagate_sse :: (image: [784] float) -> [10] float {
  using mnist_model;
  buffer1: [512] float;
  outputs: [10]  float;

  // forward propagation for layer 1.
  for i: 0..511 {
    sum: float = 0;
    image_data := image.data;
    layer1_data := layer1[i].data;
    for j: 0..195 {
      val: float = 0;
      #asm SSE {
        movups.x xmm0: vec, [image_data];
        dpps.x   xmm0, [layer1_data], 0xF1;
        movq     val, xmm0;
        add image_data, 16;
        add layer1_data, 16;
      }
      sum += val;
    }
    buffer1[i] = sum + biases1[i];
  }

  // relu for layer 1.
  buf := buffer1.data;
  for i: 0..127 {
    #asm SSE {
      movups.x xmm1: vec, [buf];
      pxor.x   xmm2: vec, xmm2;
      maxps.x  xmm1, xmm2;
      movups.x [buf], xmm1;
      add buf, 16;
    }
  }

  // forward propagation for layer 2
  for i: 0..9 {
    sum: float = 0;
    buf := buffer1.data;
    layer2_data := layer2[i].data;
    for j: 0..127 {
      val: float = 0;
      #asm SSE {
        movups.x xmm0: vec, [buf];
        dpps.x   xmm0, [layer2_data], 0xF1;
        movq     val, xmm0;
        add buf, 16;
        add layer2_data, 16;
      }
      sum += val;
    }
    outputs[i] = sum + biases2[i];
  }

  // soft max  
  return softmax(*outputs);
}

forward_propagate_cpu :: (image: [784] float) -> [10] float {
  using mnist_model;
  buffer1: [512] float;
  outputs: [10]  float;

  // forward propagation layer 1
  for i: 0..511 {
    buffer1[i] = biases1[i];
    for j: 0..783 {
      buffer1[i] += image[j] * layer1[i][j];
    }
  }

  // relu
  for i: 0..511 {
    buffer1[i] = max(0.0, buffer1[i]);
  }

  // forward propagation layer 2
  for i: 0..9 {
    outputs[i] = biases2[i];
    for j: 0..511 {
      outputs[i] += buffer1[j] * layer2[i][j];
    }
  }

  // soft max  
  return softmax(*outputs);
}

softmax :: (outputs: *[10] float) -> [10] float {
  sum : float = 0.0;
  euler : float = 2.7182818;
  outs: [] float = <<outputs;
  for i: 0..9 {
    outs[i] = Math.pow(euler, outs[i]);
    sum += outs[i];
  }

  for i: 0..9 {
    outs[i] /= sum;
    outs[i] *= 100.0;
  }
  return <<outputs;
}

MNIST_Model :: struct {
  layer1 : [512][784] float;
  biases1:      [512] float;
  layer2 :  [10][512] float;
  biases2:       [10] float;
}

mnist_model: MNIST_Model;

#import "Basic";
#import "File";
#import "String";
#import "stb_image";
Math :: #import "Math";
