package uk.ac.soton.ecs.jsh2.ml101;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;

import org.openimaj.content.slideshow.Slide;
import org.openimaj.content.slideshow.SlideshowApplication;
import org.openimaj.io.FileUtils;
import org.tartarus.snowball.ext.PorterStemmer;

import uk.ac.soton.ecs.jsh2.ml101.utils.Utils;

public class BoWDemo implements Slide {

	private HashSet<String> stopwords;

	public BoWDemo() {
		try {
			this.stopwords = new HashSet<String>();
			for (final String s : FileUtils.readlines(BoWDemo.class
					.getResourceAsStream("/org/openimaj/text/stopwords/stopwords-list.txt")))
			{
				this.stopwords.add(s);
			}
		} catch (final IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public Component getComponent(int width, int height) throws IOException {
		// the main panel
		final JPanel base = new JPanel();
		base.setOpaque(false);
		base.setPreferredSize(new Dimension(width, height));
		base.setLayout(new BoxLayout(base, BoxLayout.Y_AXIS));

		final JTextArea input = new JTextArea(200, 80);
		input.setFont(Font.decode(input.getFont().getName() + "-32"));
		input.setSize(new Dimension(width, height / 3));
		input.setPreferredSize(new Dimension(width, height / 3));
		input.setText("The quick brown fox jumped over the lazy dog");
		base.add(input);

		final JPanel controls = new JPanel();
		final JCheckBox stemming = new JCheckBox("Enable Stemmer");
		controls.add(stemming);
		final JCheckBox stopwords = new JCheckBox("Remove Stopwords");
		controls.add(stopwords);

		controls.add(new JSeparator(SwingConstants.VERTICAL));
		controls.add(new JLabel("Minimum Word Length:"));
		final JSpinner minLength = new JSpinner(new SpinnerNumberModel(0, 0, 10, 1));
		controls.add(minLength);

		final JButton button = new JButton("Extract Features");
		controls.add(button);

		base.add(controls);

		final JTextArea output = new JTextArea(200, 80);
		output.setFont(Font.decode("Monaco-28"));
		base.add(output);

		button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				final String result = extractFeaturesFormatted(input.getText(), (int) minLength.getValue(),
						stemming.isSelected(), stopwords.isSelected());
				output.setText(result);
			}
		});

		return base;
	}

	protected String extractFeaturesFormatted(String text, int minLength, boolean stem, boolean stop) {
		text = text.toLowerCase();

		TObjectIntHashMap<String> parts = new TObjectIntHashMap<String>();
		for (final String s : text.split("[^A-Z^a-z]+")) {
			parts.adjustOrPutValue(s, 1, 1);
		}

		if (stop) {
			final TObjectIntHashMap<String> toKeep = new TObjectIntHashMap<String>();
			for (final String s : parts.keySet()) {
				if (!stopwords.contains(s))
					toKeep.put(s, parts.get(s));
			}
			parts = toKeep;
		}

		if (minLength > 0) {
			final TObjectIntHashMap<String> toKeep = new TObjectIntHashMap<String>();
			for (final String s : parts.keySet()) {
				if (s.length() > minLength)
					toKeep.put(s, parts.get(s));
			}
			parts = toKeep;
		}

		if (stem) {
			final PorterStemmer stemmer = new PorterStemmer();
			final TObjectIntHashMap<String> toKeep = new TObjectIntHashMap<String>();
			for (final String s : parts.keySet()) {
				stemmer.setCurrent(s);
				if (stemmer.stem()) {
					toKeep.adjustOrPutValue(stemmer.getCurrent(), parts.get(s), parts.get(s));
				} else {
					toKeep.adjustOrPutValue(s, parts.get(s), parts.get(s));
				}
			}
			parts = toKeep;
		}

		final List<String> terms = new ArrayList<String>(parts.keySet());
		Collections.sort(terms);

		String labels = "|";
		String values = "|";
		for (final String term : terms) {
			labels += " " + term + " |";
			values += String.format(" %" + term.length() + "d |", parts.get(term));
		}

		return labels + "\n" + values;
	}

	@Override
	public void close() {
		// TODO Auto-generated method stub

	}

	public static void main(String[] args) throws IOException {
		new SlideshowApplication(new BoWDemo(), 1024, 768, Utils.BACKGROUND_IMAGE);
	}
}
