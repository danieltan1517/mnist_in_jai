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
  //data := stbi_load("MNIST/4_1.jpg", *x, *y, *comp, 0);
  data := stbi_load("MNIST/1_3.jpg", *x, *y, *comp, 0);
  buffer: [] u8;
  buffer.data = data;
  buffer.count = x * y;
  image: [784] float;

  for i: 0..783 {
    image[i] = clamp((xx buffer[i]) / 256.0, 0.0, 1.0);
  }

  results := forward_propagate_cpu(image);
  print("%\n", results);
  stbi_image_free(data);
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
  // avx512
}

forward_propagate_sse :: (image: [784] float) -> [10] float {
  using mnist_model;
  buffer1: [512] float;
  outputs: [10]  float;

  for i: 0..511 {


  }




  return outputs;
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
  sum : float = 0.0;
  euler : float = 2.7182818;
  for i: 0..9 {
    outputs[i] = Math.pow(euler, outputs[i]);
    sum += outputs[i];
  }

  for i: 0..9 {
    outputs[i] /= sum;
    outputs[i] *= 100.0;
  }
 
  return outputs;
}

MNIST_Model :: struct {
  M1 :: 784;
  N1 :: 512;
  N2 :: 10;
  M1xN1 :: M1*N1;
  N1xN2 :: N1*N2;

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
