package uk.ac.soton.ecs.jsh2.ml101;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;

import org.openimaj.content.slideshow.Slide;
import org.openimaj.content.slideshow.SlideshowApplication;
import org.openimaj.feature.FloatFVComparator;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.DisplayUtilities.ImageComponent;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourMap;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.renderer.MBFImageRenderer;
import org.openimaj.image.renderer.RenderHints;
import org.openimaj.math.geometry.line.Line2d;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.math.geometry.point.Point2dImpl;
import org.openimaj.math.geometry.shape.Circle;
import org.openimaj.math.geometry.triangulation.Voronoi;

import uk.ac.soton.ecs.jsh2.ml101.utils.Utils;

/**
 * Demo showing K-Means clustering
 *
 * @author Jonathon Hare (jsh2@ecs.soton.ac.uk)
 */
public class KMeansDemo extends MouseAdapter implements Slide, ActionListener {

	private MBFImage image;
	private ImageComponent ic;
	private BufferedImage bimg;
	private List<Point2d> points = new ArrayList<Point2d>();
	private JSpinner kSpn;
	private Point2dImpl[] centroids;
	private Float[][] colours;
	private int[] assignments;
	private JButton runBtn;
	private JButton clearBtn;
	private JButton cnclBtn;
	private volatile boolean isRunning;
	private MBFImageRenderer renderer;
	private FloatFVComparator distanceMeasure = FloatFVComparison.EUCLIDEAN;
	private JComboBox<String> distCombo;

	@Override
	public Component getComponent(int width, int height) throws IOException {
		final JPanel base = new JPanel();
		base.setOpaque(false);
		base.setPreferredSize(new Dimension(width, height));
		base.setLayout(new BoxLayout(base, BoxLayout.Y_AXIS));

		image = new MBFImage(width, height - 50, ColourSpace.RGB);
		renderer = image.createRenderer(RenderHints.ANTI_ALIASED);
		resetImage();

		ic = new DisplayUtilities.ImageComponent(true, false);
		ic.setShowPixelColours(false);
		ic.setShowXYPosition(false);
		ic.setAllowPanning(false);
		ic.setAllowZoom(false);
		ic.addMouseListener(this);
		ic.addMouseMotionListener(this);
		base.add(ic);

		final JPanel controls = new JPanel();
		controls.setPreferredSize(new Dimension(width, 50));
		controls.setMaximumSize(new Dimension(width, 50));
		controls.setSize(new Dimension(width, 50));

		clearBtn = new JButton("Clear");
		clearBtn.setActionCommand("button.clear");
		clearBtn.addActionListener(this);
		controls.add(clearBtn);

		controls.add(new JSeparator(SwingConstants.VERTICAL));
		controls.add(new JLabel("K:"));

		kSpn = new JSpinner(new SpinnerNumberModel(1, 1, 10, 1));
		controls.add(kSpn);

		controls.add(new JSeparator(SwingConstants.VERTICAL));
		controls.add(new JLabel("Distance:"));

		distCombo = new JComboBox<String>();
		distCombo.addItem("Euclidean");
		distCombo.addItem("Manhatten");
		distCombo.addItem("Cosine Distance");
		controls.add(distCombo);

		controls.add(new JSeparator(SwingConstants.VERTICAL));

		runBtn = new JButton("Run KMeans");
		runBtn.setActionCommand("button.run");
		runBtn.addActionListener(this);
		controls.add(runBtn);

		controls.add(new JSeparator(SwingConstants.VERTICAL));

		cnclBtn = new JButton("Cancel");
		cnclBtn.setEnabled(false);
		cnclBtn.setActionCommand("button.cancel");
		cnclBtn.addActionListener(this);
		controls.add(cnclBtn);

		base.add(controls);

		updateImage();

		return base;
	}

	@Override
	public void mouseClicked(MouseEvent e) {
		mouseDragged(e);
	}

	private void resetImage() {
		image.fill(RGBColour.WHITE);
		points.clear();
		assignments = null;
		centroids = null;
	}

	@Override
	public void mouseDragged(MouseEvent e) {
		if (!isRunning) {
			final Point pt = e.getPoint();
			final Point2dImpl pti = new Point2dImpl(pt.x, pt.y);
			image.drawPoint(pti, RGBColour.MAGENTA, 10);
			points.add(pti);
			updateImage();
		}
	}

	private void updateImage() {
		ic.setImage(bimg = ImageUtilities.createBufferedImageForDisplay(image, bimg));
	}

	private void initKMeans() {
		final int k = (Integer) kSpn.getValue();

		if (this.distCombo.getSelectedItem().equals("Euclidean"))
			this.distanceMeasure = FloatFVComparison.EUCLIDEAN;
		else if (this.distCombo.getSelectedItem().equals("Manhatten"))
			this.distanceMeasure = FloatFVComparison.CITY_BLOCK;
		else if (this.distCombo.getSelectedItem().equals("Cosine Distance"))
			this.distanceMeasure = FloatFVComparison.COSINE_DIST;

		this.assignments = new int[this.points.size()];
		this.centroids = new Point2dImpl[k];
		this.colours = RGBColour.coloursFromMap(ColourMap.HSV, k);

		for (int i = 0; i < k; i++) {
			centroids[i] = new Point2dImpl();
			centroids[i].x = (float) (Math.random() * image.getWidth());
			centroids[i].y = (float) (Math.random() * image.getHeight());
		}

		drawCentroidsImage(false);

	}

	private void kmeansAssignmentStep(boolean drawEvery) {
		for (int i = 0; i < points.size(); i++) {
			final Point2d pt = points.get(i);
			final int idx = assignPoint(pt);
			assignments[i] = idx;
			image.drawPoint(pt, colours[idx], 10);
			drawCentroids();

			if (drawEvery)
				updateImage();
		}
		if (!drawEvery)
			updateImage();
	}

	private boolean kmeansUpdateStep() {
		float distance = 0;

		for (int i = 0; i < centroids.length; i++) {
			final Point2dImpl oldCentroid = centroids[i];
			centroids[i] = new Point2dImpl();

			int count = 0;
			for (int j = 0; j < points.size(); j++) {
				if (assignments[j] == i) {
					centroids[i].x += points.get(j).getX();
					centroids[i].y += points.get(j).getY();
					count++;
				}
			}

			if (count == 0) {
				centroids[i].x = (float) (Math.random() * image.getWidth());
				centroids[i].y = (float) (Math.random() * image.getHeight());
			} else {
				centroids[i].x /= count;
				centroids[i].y /= count;
			}

			distance += Line2d.distance(oldCentroid, centroids[i]);

			drawCentroidsImage(true);
		}

		if (distance < 2) // 2 is a magic number!
			return true;
		return false;
	}

	private int assignPoint(Point2d pt) {
		int idx = 0;
		final float[] a = new float[2];
		final float[] b = new float[2];

		toFloatArray(pt, a);
		toFloatArray(centroids[0], b);

		float distance = (float) distanceMeasure.compare(a, b);

		for (int i = 1; i < centroids.length; i++) {
			toFloatArray(centroids[i], b);
			final float d = (float) distanceMeasure.compare(a, b);
			if (d < distance) {
				distance = d;
				idx = i;
			}
		}

		return idx;
	}

	private void toFloatArray(Point2d pt, float[] arr) {
		arr[0] = pt.getX();
		arr[1] = pt.getY();
	}

	private void drawCentroidsImage(boolean colorPoints) {
		image.fill(RGBColour.WHITE);

		for (int i = 0; i < points.size(); i++)
			image.drawPoint(points.get(i), colorPoints ? colours[assignments[i]] : RGBColour.MAGENTA, 10);

		drawCentroids();

		updateImage();
	}

	private void drawCentroids() {
		for (int i = 0; i < centroids.length; i++) {
			final Circle c = new Circle(centroids[i], 15);
			renderer.drawShapeFilled(c, colours[i]);
			renderer.drawShape(c, 3, RGBColour.BLACK);
		}
	}

	private void drawVoronoi() {
		final List<Line2d> lines = Voronoi.computeVoronoiEdges(java.util.Arrays.asList(centroids), image.getWidth(),
				image.getHeight());
		renderer.drawLines(lines, 2, RGBColour.BLACK);
		updateImage();
	}

	@Override
	public void close() {
		isRunning = false;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getActionCommand().equals("button.clear")) {
			resetImage();
			updateImage();
		} else if (e.getActionCommand().equals("button.run")) {
			runBtn.setEnabled(false);
			clearBtn.setEnabled(false);
			kSpn.setEnabled(false);
			cnclBtn.setEnabled(true);
			isRunning = true;

			new Thread(new Runnable() {
				@Override
				public void run() {
					if (isRunning)
						initKMeans();

					for (int i = 0; i < 30 && isRunning; i++) {
						if (isRunning)
							kmeansAssignmentStep(true);

						if (isRunning && kmeansUpdateStep()) {
							break;
						}
					}
					if (isRunning)
						kmeansAssignmentStep(false);

					if (isRunning)
						drawVoronoi();

					runBtn.setEnabled(true);
					clearBtn.setEnabled(true);
					kSpn.setEnabled(true);
					cnclBtn.setEnabled(false);
					isRunning = false;
				}
			}).start();
		} else if (e.getActionCommand().equals("button.cancel")) {
			isRunning = false;
			cnclBtn.setEnabled(false);
		}
	}

	public static void main(String[] args) throws IOException {
		new SlideshowApplication(new KMeansDemo(), 1024, 768, Utils.BACKGROUND_IMAGE);
	}
}
