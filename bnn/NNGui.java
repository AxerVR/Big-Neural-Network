package bnn.controller;

import jade.core.AID;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import bnn.agent.NNAgent;

/**
  @author Axel Villanueva Rodríguez.
			@TecMM Campus Zapopan
 */
public class NNGui extends JFrame {	
	private NNAgent myAgent;
	
	private JTextField inputField;
	
	public NNGui(NNAgent a) {
		super(a.getLocalName());
		
		myAgent = a;
		
		JPanel p = new JPanel();
		p.setLayout(new GridLayout(10, 2));
		p.add(new JLabel("Neural Network:"));
		p.add(new JLabel(""));
		p.add(new JLabel("Input: "));
		inputField = new JTextField(15);
		p.add(inputField);
		getContentPane().add(p, BorderLayout.CENTER);
		
		JButton classifyBtn = new JButton("Classify");
		classifyBtn.addActionListener( new ActionListener() {
			public void actionPerformed(ActionEvent ev) {
				try {
					String input = inputField.getText().trim();
					if(!input.equals("")) myAgent.btnAction(input);
					else System.out.println("Empty input.");
					inputField.setText("");
				}
				catch (Exception e) {
					JOptionPane.showMessageDialog(NNGui.this, "Invalid values. " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE); 
				}
			}
		} );
		
		p = new JPanel();
		p.add(classifyBtn);
		getContentPane().add(p, BorderLayout.SOUTH);
		
		// Make the agent terminate when the user closes 
		// the GUI using the button on the upper right corner	
		addWindowListener(new	WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				myAgent.doDelete();
			}
		} );
		
		setResizable(false);
	}
	
	public void showGui() {
		pack();
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		int centerX = (int)screenSize.getWidth() / 2;
		int centerY = (int)screenSize.getHeight() / 2;
		setLocation(centerX - getWidth() / 2, centerY - getHeight() / 2);
		super.setVisible(true);
	}	
}
