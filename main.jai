main :: () {
  win := create_window(800, 600, "Window");
  window_width, window_height := simp.get_render_dimensions(win);
  simp.set_render_target(win);
  ui_init();
  setup_getrect_theme(.Grayscale);
  init_mnist_images();
  while eventloop := true {
    Input.update_window_events();
    for Input.get_window_resizes() {
      simp.update_window(it.window);
      if it.window == win {
        window_width  = it.width;
        window_height = it.height;
      }
    }

    mouse_pressed := false;
    for event: Input.events_this_frame {
      if event.type == .QUIT then {
        break eventloop;
      }
      getrect_handle_event(event);
    }

    current_time := get_time();
    render(win, current_time);
    sleep_milliseconds(10);
    reset_temporary_storage();
  }
}

Image :: struct {
  picture: *simp.Texture;
  image_data: [784] float;
}

MNIST_images: [50] Image;

init_mnist_images :: () {
  index := 0;
  print("MNIST Images ");
  for i: 0..9 {
    for j: 1..5 {
      file_name := tprint("MNIST/%1_%2.jpg", i, j);
      success := load_image(index, file_name);
      assert(success, "Error. File % failed.\n", file_name);
      print("[%] ", file_name);
      index += 1;
    }
  }
  print("loaded.\n");

  load_image :: (index: int, filename: string) -> bool {
    // TODO: there should be a better way than loading the
    // same image twice.
    // NOTE: need to call simp.set_render_target before calling this function.
    picture := New(simp.Texture);
    success := simp.texture_load_from_file(picture, filename);
    print("%\n", <<picture);
    MNIST_images[index].picture = picture;
    if !success then
      return false;

    x, y, comp: s32;
    data := stbi_load(to_c_string(filename), *x, *y, *comp, 0);
    defer stbi_image_free(data);
    buffer: [] u8;
    buffer.data = data;
    buffer.count = x * y;
    image := *MNIST_images[index].image_data;
    assert(buffer.count == 784, "Error. Image has incorrect dimensions [%1x%2].\n", x, y);
    for i: 0..783 {
      grey := (xx buffer[i]) / 256.0;
      (<<image)[i] = grey;
    }
    return true;
  }
}

rindex := 0;

render :: (win: Window_Type, current_time: float64) #expand {
  // background.
  simp.clear_render_target(.05, .05, .05, 1);
  simp.immediate_begin();
  
  defer {
    simp.swap_buffers(win);
    simp.immediate_flush();
  }

  width, height := simp.get_render_dimensions(win);
  
  texture := MNIST_images[rindex].picture;
  simp.set_shader_for_images(texture);

  color := make_vector4(1.0, 1.0, 1.0, 1.0);
  
  simp.immediate_quad(100, 100, 250, 250, color);

  // update ui
  ui_per_frame_update(win, width, height, current_time);
  k := clamp(height * .05, 17.0, 30.0);
  r := get_rect(250.0, (xx height/2) - 5.0 - k, 6.5*k, k);
  if button(r, "Previous") {
    rindex -= 1;
    if rindex < 0 then {
      rindex = 49;
    }
  }

  r.y -= k + 5.0;
  if button(r, "Next") {
    rindex += 1;
    if rindex > 49 then {
      rindex = 0;
    }
  }

  r.y -= k + 5.0;
  if button(r, "Random") {
    rseed ^= rseed >> 12;
    rseed ^= rseed << 25;
    rseed ^= rseed >> 27;
    rseed *= 2685821657736338717;
    rindex = Math.abs(rseed) % 50;
  }
}

rseed : int = 861097715;

setup_getrect_theme :: (theme: Default_Themes) #expand {
  proc := default_theme_procs[theme];
  getrect_theme = proc();
  button_theme := *getrect_theme.button_theme;
  button_theme.label_theme.alignment = .Left;

  slider_theme := *getrect_theme.slider_theme;
  slider_theme.foreground.alignment = .Left;
  set_default_theme(getrect_theme);
}

getrect_theme: Overall_Theme; 

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
#import "Math";
#import "stb_image";
Math :: #import "Math";
simp :: #import "Simp";
Input :: #import "Input";
#import "Window_Creation";
#import "GetRect";
