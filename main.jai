main :: () {
  win := create_window(800, 600, "MNIST in Jai");
  window_width, window_height := simp.get_render_dimensions(win);
  simp.set_render_target(win);
  ui_init();
  setup_getrect_theme(.Grayscale);
  init_mnist_images();
  forward_propagate(MNIST_images[rindex].image_data);
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
    render(win, window_width, window_height, current_time);
    sleep_milliseconds(10);
    reset_temporary_storage();
  }
}

#scope_file

Image :: struct {
  texture: simp.Texture;
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
    // Jon said to fill out a "Bitmap" struct, and then use "texture_load_from_bitmap"
    bitmap: simp.Bitmap;
    success := simp.bitmap_load(*bitmap, filename);
    if !success then
      return false;
    image := *MNIST_images[index].image_data;
    assert(bitmap.data.count == 784, "Error. Image has incorrect dimensions [%1x%2].\n", bitmap.width, bitmap.height);
    for i: 0..783 {
      grey := (xx bitmap.data[i]) / 256.0;
      (<<image)[i] = grey;
    }

    copy_data := bitmap.data;
    defer array_free(copy_data);
    bitmap.data = NewArray(2352, u8); // 2352 <- 784 x 3
    for i: 0..783 {
      byte := copy_data[i];
      bitmap.data[i*3 + 0] = byte;
      bitmap.data[i*3 + 1] = byte;
      bitmap.data[i*3 + 2] = byte;
    }
    bitmap.format = .RGB8;

    texture := *MNIST_images[index].texture;
    success = simp.texture_load_from_bitmap(texture, *bitmap);
    return true;
  }
}

rindex := 0;
best_guess_number := 0;
best_probability := 0.0;

render :: (win: Window_Type, width: s32, height: s32, current_time: float64) {
  // background.
  simp.clear_render_target(.10, .10, .10, 1);
  simp.immediate_begin();
  k := clamp(height * .05, 17.0, 30.0);
  fwidth := cast(float) width;
  fheight := cast(float) height;
  cenx := fwidth / 2.0;
  ceny := fheight / 2.0;
  
  defer {
    simp.swap_buffers(win);
    simp.immediate_flush();
  }
  
  texture := *MNIST_images[rindex].texture;
  simp.set_shader_for_images(texture);

  sq := min(cenx,ceny) * .95 - k - k - 7;

  color := make_vector4(1.0, 1.0, 1.0, 1.0);
  simp.immediate_quad(cenx - sq, ceny - sq, cenx + sq, ceny + sq, color);

  // update ui
  ui_per_frame_update(win, width, height, current_time);
  r := get_rect(cenx - sq, ceny + sq + 3.5, sq*.5, k);
  if button(r, "Previous") {
    rindex -= 1;
    if rindex < 0 then {
      rindex = 49;
    }
    forward_propagate(MNIST_images[rindex].image_data);
  }

  r.x += sq*.5;
  if button(r, "Next") {
    rindex += 1;
    if rindex > 49 then {
      rindex = 0;
    }
    forward_propagate(MNIST_images[rindex].image_data);
  }

  r.x += sq*.5;
  if button(r, "Random") {
    rseed ^= rseed >> 12;
    rseed ^= rseed << 25;
    rseed ^= rseed >> 27;
    rseed *= 2685821657736338717;
    rindex = Math.abs(rseed) % 50;
    forward_propagate(MNIST_images[rindex].image_data);
  }

  r.x += sq*.5;
  calc := calculate;
  dropdown(r, .["sse", "cpu"], *calc);
  r = get_rect(cenx - sq, ceny - sq - k - 3.5, sq*2, k);
  str := tprint("%1 with %2%% Probability", best_guess_number, formatFloat(best_probability, zero_removal=.NO, trailing_width=2));
  label(r, str, label_theme);

  r.y -= k;
  str = tprint("Execution Time: % ms", formatFloat(1000*exe_time, zero_removal=.NO,trailing_width=2));
  label(r, str, label_theme);

  r = get_rect(cenx + sq, ceny + sq - k - 3.5, sq*2, k);
  label_color := label_theme.text_color;
  for i: 0..9 {
    val := probabilities[i];
    if best_guess_number == i {
      label_theme.text_color = make_vector4(1,1,1,1);
    }
    str := tprint("%1: %2%%", i, formatFloat(val, zero_removal=.NO, trailing_width=2));
    label(r, str, label_theme);
    label_theme.text_color = label_color;
    r.y -= k;
  }

  draw_popups();
  if calc != calculate {
    calculate = calc;
    forward_propagate(MNIST_images[rindex].image_data);
  }
}

calculate: s32 = 0;
rseed : int = 861097715;

setup_getrect_theme :: (theme: Default_Themes) #expand {
  proc := default_theme_procs[theme];
  getrect_theme = proc();
  label_theme = *getrect_theme.label_theme;
  label_theme.alignment = .Left;
  set_default_theme(getrect_theme);
}

getrect_theme: Overall_Theme;
label_theme: *Label_Theme;

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

probabilities: [10] float;
exe_time: float64;
forward_propagate :: (image: [784] float) {
  exe_time = get_time();
  if calculate == 0 then {
    probabilities = forward_propagate_sse(image);
  } else if calculate == 1 then {
    probabilities = forward_propagate_cpu(image);
  }

  best_guess_number = -1;
  best_probability = -100.0;
  for prob, index: probabilities {
    if prob > best_probability {
      best_guess_number = index;
      best_probability = prob;
    }
  }
  exe_time = get_time() - exe_time;
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

#no_reset mnist_model: MNIST_Model;
#run {
  #import "Compiler";
  success := load_neural_network(file_name = "mnist.nn");
  assert(success);
  set_build_options_dc (.{
    output_executable_name = "mnist_in_jai",
  });
}

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
#import "Machine_X64";
