// main71.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: fastjet; jet finding;

// Simple example of fastjet analysis. Roughly follows analysis of:
// T. Aaltonen et al. [CDF Collaboration],
// Measurement of the cross section for W-boson production in association
// with jets in ppbar collisions at sqrt(s)=1.96$ TeV
// Phys. Rev. D 77 (2008) 011108
// arXiv:0711.4044 [hep-ex]
//
// Cuts:
//   ET(elec)     > 20GeV
//   |eta(elec)|  < 1.1
//   ET(missing)  > 30GeV
//   ET(jet)      > 20GeV
//   |eta(jet)|   < 2.0
//   deltaR(elec, jet) > 0.52
// Not used:
//   mT(W)        > 20GeV
//

#include "Pythia8/Pythia.h" 

// This is the minimal interface needed to access FastJet.
// A more sophisticated interface is demonstrated in main72.cc.
#include "Pythia8Plugins/FastJet3.h"
#include "fastjet/ClusterSequenceArea.hh"
#include "fjcontrib/RecursiveTools/SoftDrop.hh"
#include "TH1F.h"
#include "TFile.h" 
#include "TH2F.h"
#include "TCanvas.h"
#include "main72.h"
#include "TPad.h"

#include <chrono>
#include <iostream>
#include <fstream>


using namespace Pythia8;

int main() {
  // Number of events, generated and listed ones (for jets).
  int nEvent    = 5000;
 
  // Select common parameters for SlowJet and FastJet analyses.
  int    power   = -1;     // -1 = anti-kT; 0 = C/A; 1 = kT.
  double R       = 0.4;    // Jet size.
  double pTMin   = 5.0;    // Min jet pT.
  double etaMax  = 2.5;    // Pseudorapidity range of detector.
  int    select  = 2;      // Which particles are included?
  int    massSet = 2;      // Which mass are they assumed to have?

  //My parameters
  double alpha=0.1; //Q-jet parameter
  int power2 = 0;
  float z_cut = 0.1;
  int beta = 1;
  float lowlim = 200;  
  float uplim = 300;
  int bins = 20;
  float a = 1.;
  vector <int> pow_l = {0};//,1,-1};

  // Generator. Shorthand for event.
  Pythia pythia;
  Event& event = pythia.event;

  //Pythia settings
  ostringstream pythiaset;
  pythiaset << "pythia_settings.cmd";
  pythia.readFile(pythiaset.str());
  pythia.init(); 

  // Set up FastJet jet finder.
  double ghost_area=0.01;
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm , R, power); 
  fastjet::AreaDefinition areaDef(fastjet::active_area_explicit_ghosts, fastjet::GhostedAreaSpec(2.5, 1, ghost_area));
  
  std::vector <fastjet::PseudoJet> fjInputs; 


  //Make histograms and root file
  TFile *file = TFile::Open("pt_spectr_5_R0.4.root","recreate");
  TH1F* h1 = new TH1F("h1", "p_{t}  spectrum", 50, 0., 2000.);
    
  //Without background
  vector <TH1F*> list_hist_1 = MakeListHistSpecter(bins, "1");
  vector <TH1F*> list_hist_2 = MakeListHistSpecter(bins, "2");
  vector <TH1F*> list_hist_3 = MakeListHistSpecter(bins, "3");

  vector <vector <TH1F*>> list_hist_a = {list_hist_1, list_hist_2, list_hist_3};

  //With background
  vector <TH1F*> list_hist_4 = MakeListHistSpecter(bins, "4");
  vector <TH1F*> list_hist_5 = MakeListHistSpecter(bins, "5");
  vector <TH1F*> list_hist_6 = MakeListHistSpecter(bins, "6");

  vector <vector <TH1F*>> list_hist_b = {list_hist_4, list_hist_5, list_hist_6};

  //With background2 
  vector <TH1F*> list_hist_7 = MakeListHistSpecter(bins, "7");
  vector <TH1F*> list_hist_8 = MakeListHistSpecter(bins, "8");
  vector <TH1F*> list_hist_9 = MakeListHistSpecter(bins, "9");

  vector <vector <TH1F*>> list_hist_c = {list_hist_7, list_hist_8, list_hist_9};

  vector <vector <vector <TH1F*>>> list_hist = {list_hist_a, list_hist_b, list_hist_c};

  
  
  TH2F* h2D =  new TH2F("h2D", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);
  TH2F* h2D2 =  new TH2F("h2D2", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);
  TH2F* h2D3 =  new TH2F("h2D3", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);

  vector <TH2F*> list_2d_1 = {h2D, h2D2, h2D3};

   
  TH2F* h2D4 =  new TH2F("h2D4", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);
  TH2F* h2D5 =  new TH2F("h2D5", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);
  TH2F* h2D6 =  new TH2F("h2D6", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);

  vector <TH2F*> list_2d_2 = {h2D4, h2D5, h2D6};

   
  TH2F* h2D7 =  new TH2F("h2D7", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);
  TH2F* h2D8 =  new TH2F("h2D8", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);
  TH2F* h2D9 =  new TH2F("h2D9", "Primary Lund Plane", 50, 0., 10., 50, -12., 0.);

  vector <TH2F*> list_2d_3 = {h2D7, h2D8, h2D9};

  vector <vector <TH2F*>> list_2d = {list_2d_1, list_2d_2, list_2d_3};
  
  

  auto start = std::chrono::high_resolution_clock::now();
  // Begin event loop. Generate event. Skip if error.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (!pythia.next()) continue;
    // Begin FastJet analysis: extract particles from event record.
    fjInputs.resize(0);
    for (int i = 0; i < event.size(); ++i) if (event[i].isFinal()) {

      // Require visible/charged particles inside detector.
      if      (select > 2 &&  event[i].isNeutral() ) continue;
      else if (select == 2 && !event[i].isVisible() ) continue;
      if (etaMax < 20. && abs(event[i].eta()) > etaMax) continue;

      // Create a PseudoJet from the complete Pythia particle.
      fastjet::PseudoJet particleTemp = event[i];

      // Store acceptable particles as input to Fastjet.
      fjInputs.push_back( particleTemp);
    }
 
    // Run Fastjet algorithm and sort jets in pT order.
    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequence clustSeq(fjInputs, jetDef);
    inclusiveJets = clustSeq.inclusive_jets(pTMin);
    sortedJets    = sorted_by_pt(inclusiveJets);
    //vector <fastjet::PseudoJet> particles = sortedJets[0].constituents();
    if (sortedJets.size()==0) continue;
    if (sortedJets[0].constituents().size() < 2) continue;
    //if (sortedJets[0].pt()<lowlim || sortedJets[0].pt()>uplim) continue;

    //Without background
    double w = pythia.info.weight();
    h1->Fill(sortedJets[0].pt(),w);

    //With background
    vector <fastjet::PseudoJet> bgparticles = getbackground(750, 0.52, 120, 102.5);
    vector <fastjet::PseudoJet> bgparticles2 = getbackground(5000, 0.63, 100, 77.6595);

    //Make vector with particles
    vector <vector <fastjet::PseudoJet>> particle_list = {sortedJets[0].constituents(), GetParticleList(fjInputs, bgparticles, jetDef, areaDef,pTMin, R),GetParticleList(fjInputs, bgparticles2, jetDef, areaDef, pTMin, R) };

    //Start loop for this event
    for (int j=0; j<3; ++j){ 
      //for the 3 different particle lists
  
      for (int i=0; i<1; ++i){  
        //For the 3 different algorithms

        //cluster
        jetDef.set_extra_param(pow_l[i]); 
        vector <fastjet::PseudoJet> iJ, sJ;
        fastjet::ClusterSequence clustSeq2(particle_list[j], jetDef);  
        iJ = clustSeq2.inclusive_jets(pTMin);
        sJ    = sorted_by_pt(iJ);
        if (sJ.size()==0) continue; //Counteract segmentation fault (unknown)
        if (sJ[0].constituents().size() <2) continue; //Counteract segmentation fault (less than two exclusive subjets)

        //Make observables after applying algorithm only
        vector <fastjet::PseudoJet> listjet = sJ[0].exclusive_subjets(2);
        float theta = listjet[0].delta_R(listjet[1]);
        auto [mh, m0, z, kt] = get_m(listjet);
        float N_tot = sJ[0].constituents().size();
        float N1 = listjet[0].constituents().size();
        float N2 = listjet[1].constituents().size();
        //Plot observables
        list_hist[j][i][0]->Fill(N_tot, w);
        list_hist[j][i][1]->Fill(theta/R, w);
        list_hist[j][i][2]->Fill(sqrt(mh)/sJ[0].pt(),w);
        list_hist[j][i][3]->Fill(sqrt(kt)/sJ[0].pt());
        list_hist[j][i][4]->Fill(z, w);
        list_hist[j][i][5]->Fill(N1, w);
        list_hist[j][i][6]->Fill(N2, w);
        list_hist[j][i][7]->Fill((N1-N2)/N_tot, w);

        //SoftDrop
        auto[sdjet,nrSD] = SoftDrop2({sJ[0]}, z_cut, beta, R);
        std::vector <fastjet::PseudoJet > sdlistjet	= sdjet[0].exclusive_subjets(2);
        float sdtheta = sdlistjet[0].delta_R(sdlistjet[1]);
        auto [sdmh, sdm0, sdz, sdkt] = get_m(sdlistjet);
        float sdN_tot = sdjet[0].constituents().size();
        float sdN1 = sdlistjet[0].constituents().size();
        float sdN2 = sdlistjet[1].constituents().size();
        //Plot SD observables
        list_hist[j][i][8]->Fill(sdN_tot,w);
        list_hist[j][i][9]->Fill(nrSD, w);
        list_hist[j][i][10]->Fill(sdtheta/R, w);
        list_hist[j][i][11]->Fill(sqrt(sdmh)/sdjet[0].pt(),w);
        list_hist[j][i][12]->Fill(sqrt(sdkt)/sdjet[0].pt(), w);
        list_hist[j][i][13]->Fill(sdz, w);
        list_hist[j][i][14]->Fill(sdN1, w);
        list_hist[j][i][15]->Fill(sdN2, w); 
        list_hist[j][i][16]->Fill((sdN1-sdN2)/sdN_tot, w);

        //Dynamical Grooming
        /*
        fastjet::PseudoJet hardest_split = dynamicalGrooming(sJ[0], R, a);
        vector <fastjet::PseudoJet> hardlist = hardest_split.exclusive_subjets(2);
        float dgtheta = hardlist[0].delta_R(hardlist[1]);
        auto [dgmh, dgm0, dgz, dgkt] = get_m(hardlist);
        float dgN_tot = hardest_split.constituents().size();
        float dgN1 = hardlist[0].constituents().size();
        float dgN2 = hardlist[1].constituents().size();
        //Plot DG observables
        list_hist[j][i][17]->Fill(dgN_tot,w);
        list_hist[j][i][18]->Fill(dgtheta/R, w);
        list_hist[j][i][19]->Fill(sqrt(dgmh)/hardest_split.pt(),w);
        list_hist[j][i][20]->Fill(sqrt(dgkt)/hardest_split.pt(), w);
        list_hist[j][i][21]->Fill(dgz, w);
        list_hist[j][i][22]->Fill(dgN1, w);
        list_hist[j][i][23]->Fill(dgN2, w);
        list_hist[j][i][24]->Fill((dgN1-dgN2)/dgN_tot, w);
        */

        
        if (pow_l[i] == 0){
          if (sdN_tot > 13) continue;
          list_hist[j][i][25]->Fill(sdN_tot,w);
          list_hist[j][i][26]->Fill(nrSD, w);
          list_hist[j][i][27]->Fill(sdtheta/R, w);
          list_hist[j][i][28]->Fill(sqrt(sdmh)/sdjet[0].pt(),w);
          list_hist[j][i][29]->Fill(sqrt(sdkt)/sdjet[0].pt(), w);
          list_hist[j][i][30]->Fill(sdz, w);
          list_hist[j][i][31]->Fill(sdN1, w);
          list_hist[j][i][32]->Fill(sdN2, w); 
          list_hist[j][i][33]->Fill((sdN1-sdN2)/sdN_tot, w);
          LundPlane(sdjet[0], list_2d[j][2], w, R);
        }

        //Lund plane stuffs
        LundPlane(sortedJets[0], list_2d[j][0], w, R);
        LundPlane(sdjet[0], list_2d[j][1], w, R);
        
        //end
        
      }
    }



    /*

    
    */

  }
  //Event loop done  
  
  cout << std::setprecision(10);

  float truepos = list_hist[0][0][25]->Integral();
  float falseneg = list_hist[0][0][8]->Integral() - list_hist[0][0][25]->Integral();
  float falsepos = list_hist[1][0][25]->Integral();
  float trueneg =  list_hist[1][0][8]->Integral() - list_hist[1][0][25]->Integral();

  float trueposrate = truepos/(truepos+falseneg);
  float falseposrate = falsepos/(falsepos+trueneg);

  cout << "True positive rate = " << trueposrate << endl;
  cout << "False positive rate = " << falseposrate << endl;

  //Plot pt specter
  PlotSpectr("pt", h1, 1, "p_{t}", "Original anti-kt", " ");

  //List of canvas names 
  vector <string> list_can_1 = {"Ntot", "theta", "mh", "kt", "z", "N1", "N2", "N1-N2", "SD_Ntot", "SD_ng","SD_theta", "SD_mh", "SD_kt", "SD_z", "SD_N1", "SD_N2", "SD_N1-N2","DG_Ntot", "DG_theta", "DG_mh", "DG_kt", "DG_z", "DG_N1", "DG_N2", "DG_N1-N2", "cut_Ntot", "cut_nG","cut_theta", "cut_mh", "cut_kt", "cut_z ", "cut_N1", "cut_N2", "cut_N1-N2"};
  vector <string> list_can_2 = {"kt_Ntot", "kt_theta", "kt_mh", "kt_kt", "kt_z", "kt_N1", "kt_N2", "kt_N1-N2", "kt_SD_Ntot", "kt_SD_ng", "kt_SD_theta", "kt_SD_mh", "kt_SD_kt", "kt_SD_z", "kt_SD_N1", "kt_SD_N2", "kt_SD_N1-N2","kt_DG_Ntot", "kt_DG_theta", "kt_DG_mh", "kt_DG_kt", "kt_DG_z", "kt_DG_N1", "kt_DG_N2", "kt_DG_N1-N2"};
  vector <string> list_can_3 = {"antikt_Ntot", "antikt_theta", "antikt_mh", "antikt_kt", "antikt_z", "antikt_N1", "antikt_N2", "antikt_N1-N2", "antikt_SD_Ntot","antikt_SD_ng", "antikt_SD_theta", "antikt_SD_mh", "antikt_SD_kt", "antikt_SD_z", "antikt_SD_N1", "antikt_SD_N2", "antikt_SD_N1-N2","antikt_DG_Ntot", "antikt_DG_theta", "antikt_DG_mh", "antikt_DG_kt", "antikt_DG_z", "antikt_DG_N1", "antikt_DG_N2", "antikt_DG_N1-N2"};
  vector <vector <string>> list_can = {list_can_1, list_can_2, list_can_3};
 
  //List of label names
  vector <string> list_lab = {"N_{tot}", "#theta/R", "m_{h}/p_{t}", "k_{t}/p_{t}", "z", "N_{1}", "N_{2}", "N_{1}-N_{2}/N_{tot}", "N_{tot}", "n_{g}","#theta/R", "m_{h}/p_{t}", "k_{t}/p_{t}", "z", "N_{1}", "N_{2}", "N_{1}-N_{2}/N_{tot}","N_{tot}", "#theta/R", "m_{h}/p_{t}", "k_{t}/p_{t}", "z", "N_{1}", "N_{2}", "N_{1}-N_{2}/N_{tot}","N_{tot}", "n_{G}","#theta/R", "m_{h}/p_{t}", "k_{t}/p_{t}", "z", "N_{1}", "N_{2}", "N_{1}-N_{2}/N_{tot}"};
  vector <string> list_type = {"C/A", "k_{t}", "Anti-k_{t}"};
  vector <string> wbg = {"without background", "with background 1", "with background 2"}; 
    
  //Draw plots  d        
  for (int i=0; i<1; ++i){  
    for (int k=0; k<list_can[i].size(); ++k){
      PlotSpectr2(list_can[i][k], list_hist[0][i][k], list_hist[1][i][k],list_hist[2][i][k], bins, list_lab[k], list_type[i], wbg);
    }
  }
  
  

  //Plot Lund plane stuff
  for (int j=0; j<3; ++j){
    TCanvas *c2D = new TCanvas(Form("c2D_%i",j),Form("c2D_%i",j), 10, 5, 1000, 500);
    c2D->Draw();
    c2D->cd();
    TPad *myPad1 = new TPad("myPad1", "myPad1", 0, 0, 0.33, 1.0);
    myPad1->Draw();
    myPad1->cd();
    list_2d[j][0]->SetStats(0);
    list_2d[j][0]->GetYaxis()->SetTitle("ln kt/Rpt");
    list_2d[j][0]->GetXaxis()->SetTitle("ln R/#theta");
    list_2d[j][0]->SetTitle("No Grooming");
    list_2d[j][0]->Draw("COL");

    c2D->cd();
    TPad *myPad2 = new TPad("myPad2", "myPad2", 0.33, 0, 0.66, 1.0);
    myPad2->Draw();
    myPad2->cd();
    list_2d[j][1]->SetStats(0);
    list_2d[j][1]->GetYaxis()->SetTitle("ln kt/Rpt");
    list_2d[j][1]->GetXaxis()->SetTitle("ln R/#theta");
    list_2d[j][1]->SetTitle("SoftDrop");
    list_2d[j][1]->Draw("COL");
    
    c2D->cd();
    TPad *myPad3 = new TPad("myPad3", "myPad3", 0.66, 0, 0.99, 1.0);
    myPad3->Draw();
    myPad3->cd();
    list_2d[j][2]->SetStats(0);
    list_2d[j][2]->GetYaxis()->SetTitle("ln kt/Rpt");
    list_2d[j][2]->GetXaxis()->SetTitle("ln R/#theta");
    list_2d[j][2]->SetTitle("SoftDrop & cut");
    list_2d[j][2]->Draw("COL");
    //c2D->SaveAs("pad.png");
    c2D->Write();
  }



  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(stop - start);
  cout << "Time it takes = " << duration.count()*pow(10,-6) << " s" << endl;

  delete file;

  return 0;
}

//cut -0.6: TPR=0.38, FPR=0.19
//cut -0.5: TPR=0.45, FPR=0.27
//cut -0.4: TPR=0.58, FPR=0.35
//cut -0.3: TPR=0.61, FPR=0.44


