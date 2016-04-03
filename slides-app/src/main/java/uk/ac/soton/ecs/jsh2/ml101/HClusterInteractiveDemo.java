package uk.ac.soton.ecs.jsh2.ml101;

import java.io.IOException;

import javax.swing.JSplitPane;

import org.openimaj.content.slideshow.SlideshowApplication;

import uk.ac.soton.ecs.jsh2.ml101.utils.GroovyREPLConsoleSlide;
import uk.ac.soton.ecs.jsh2.ml101.utils.Utils;

public class HClusterInteractiveDemo extends GroovyREPLConsoleSlide {

	public HClusterInteractiveDemo() throws IOException {
		super(JSplitPane.VERTICAL_SPLIT, HClusterInteractiveDemo.class.getResource("hcluster.groovy"),
				"result = hcluster(moduledata.transpose())", "result = hcluster(moduledata)");
	}

	public static void main(String[] args) throws IOException {
		new SlideshowApplication(new HClusterInteractiveDemo(), 1024, 768, Utils.BACKGROUND_IMAGE);
	}
}
