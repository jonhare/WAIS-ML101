package uk.ac.soton.ecs.jsh2.ml101.utils;

import java.awt.Cursor;
import java.awt.Desktop;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.net.URI;

import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.UIManager;

public class Utils {
	public static BufferedImage BACKGROUND_IMAGE = null;
	static {
		// try {
		BACKGROUND_IMAGE = new BufferedImage(1024, 768, BufferedImage.TYPE_INT_ARGB);
		final Graphics2D g = BACKGROUND_IMAGE.createGraphics();
		g.setColor(UIManager.getColor("Panel.background"));
		g.fillRect(0, 0, BACKGROUND_IMAGE.getWidth(), BACKGROUND_IMAGE.getHeight());
		// } catch (final IOException e) {
		// }
	}

	private Utils() {
	}

	public static JLabel linkify(final String text, String URL, String toolTip)
	{
		URI temp = null;
		try
		{
			temp = new URI(URL);
		} catch (final Exception e)
		{
			e.printStackTrace();
		}
		final URI uri = temp;
		final JLabel link = new JLabel();
		link.setText("<HTML><FONT color=\"#000099\">" + text + "</FONT></HTML>");
		if (!toolTip.equals(""))
			link.setToolTipText(toolTip);
		link.setCursor(new Cursor(Cursor.HAND_CURSOR));
		link.addMouseListener(new MouseListener()
		{
			@Override
			public void mouseExited(MouseEvent arg0)
			{
				link.setText("<HTML><FONT color=\"#000099\">" + text + "</FONT></HTML>");
			}

			@Override
			public void mouseEntered(MouseEvent arg0)
			{
				link.setText("<HTML><FONT color=\"#000099\"><U>" + text + "</U></FONT></HTML>");
			}

			@Override
			public void mouseClicked(MouseEvent arg0)
			{
				if (Desktop.isDesktopSupported())
				{
					try
					{
						Desktop.getDesktop().browse(uri);
					}
					catch (final Exception e)
					{
						e.printStackTrace();
					}
				}
				else
				{
					final JOptionPane pane = new JOptionPane("Could not open link.");
					final JDialog dialog = pane.createDialog(new JFrame(), "");
					dialog.setVisible(true);
				}
			}

			@Override
			public void mousePressed(MouseEvent e)
			{
			}

			@Override
			public void mouseReleased(MouseEvent e)
			{
			}
		});
		return link;
	}
}
