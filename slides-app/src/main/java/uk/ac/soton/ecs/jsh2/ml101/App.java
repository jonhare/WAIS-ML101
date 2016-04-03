package uk.ac.soton.ecs.jsh2.ml101;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.content.slideshow.PictureSlide;
import org.openimaj.content.slideshow.Slide;
import org.openimaj.content.slideshow.SlideshowApplication;

import uk.ac.soton.ecs.jsh2.ml101.utils.Utils;

public class App {
	public static void main(String[] args) throws IOException {
		final List<Slide> slides = new ArrayList<Slide>();

		for (int i = 1; i <= 45; i++) {
			slides.add(new PictureSlide(App.class.getResource(String.format("slides.%03d.jpeg", i))));
		}

		slides.set(10, new SimpleMeanColourFeatureDemo());
		slides.set(11, new BoWDemo());

		slides.set(24, new LinearClassifierDemo());
		slides.set(30, new KNNDemo());

		slides.set(36, new KMeansDemo());
		slides.set(40, new HClusterDemo());
		slides.set(42, new HClusterInteractiveDemo());
		slides.set(43, new KMeansInteractiveDemo());

		new SlideshowApplication(slides, 1024, 768, Utils.BACKGROUND_IMAGE);
	}
}
