import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;

int drawRadius = 60;
int bgcolor = 30;

Path chaiPath;
Path imageSavePath;

void setup() {
  size(1000,1000);
  ellipseMode(RADIUS);
  background(bgcolor);
  fill(255);
  strokeWeight(drawRadius);
  stroke(255);
  println(System.getProperty("user.dir"));
  println(Paths.get(".").toAbsolutePath().normalize().toString());
  println(Path.of("").toAbsolutePath().toString());
  println(this.getClass().getClassLoader().getResource("").getPath().toString());
  chaiPath = Paths.get(sketchPath());//.getParent().getParent();
  imageSavePath = chaiPath.resolve("number.png");
  println(chaiPath.toString());
  println(imageSavePath.toString());
}

void draw() {
  if (mousePressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}


void keyPressed() {
  if (key == ' ') {
    save(imageSavePath.toString());
  }
  if (key == 'c') {
    background(bgcolor);
  }
}
