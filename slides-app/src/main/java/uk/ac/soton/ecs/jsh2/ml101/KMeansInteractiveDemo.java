package uk.ac.soton.ecs.jsh2.ml101;

import java.io.IOException;

import javax.swing.JSplitPane;

import org.openimaj.content.slideshow.SlideshowApplication;

import uk.ac.soton.ecs.jsh2.ml101.utils.GroovyREPLConsoleSlide;
import uk.ac.soton.ecs.jsh2.ml101.utils.Utils;

public class KMeansInteractiveDemo extends GroovyREPLConsoleSlide {

	public KMeansInteractiveDemo() throws IOException {
		super(JSplitPane.VERTICAL_SPLIT, HClusterInteractiveDemo.class.getResource("kmeanscluster.groovy"),
				"prettyPrint(kmeans(moduledata.transpose(),20).items)",
				"prettyPrint(result.topFeatures)",
				"prettyPrint((result=kmeans(moduledata,20)).items)");
	}

	public static void main(String[] args) throws IOException {
		new SlideshowApplication(new KMeansInteractiveDemo(), 1024, 768, Utils.BACKGROUND_IMAGE);
	}
}
