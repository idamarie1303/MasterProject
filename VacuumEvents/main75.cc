// main75.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Nishita Desai <nishita.desai@tifr.res.in>

// Keywords: jet finding; fastjet; BSM; dark matter;

// This is a simple test program to study jets in Dark Matter production.

#include "Pythia8/Pythia.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include <math.h>
#include <boost/math/special_functions/bessel.hpp>
#include <random>
#include "TH1F.h"
#include "TGraph.h"
#include "TFile.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TLegend.h"
#include <iostream>
#include <fstream>

#include "TGraphAsymmErrors.h"


using namespace Pythia8;

double_t myFunc(float p_T, float beta_T, float T_F, float A_i) { 
  float rho = atanh(beta_T);
  float inK = p_T*cosh(rho) / T_F;
  float inI = p_T*sinh(rho) / T_F;

  float I0 = boost::math::cyl_bessel_i(0, inI);
  float K1 = boost::math::cyl_bessel_k(1, inK);

  return A_i*p_T*K1*I0;
}

int main() {
  

  




  
  

  fastjet::PseudoJet kjet = fastjet::PseudoJet(20.*cos(0.5), 20.0*sin(0.5), (20.+15.)*sinh(1.5), (20.+15.)*cosh(1.5));
  cout << kjet.pt() << endl;

  
  float beta_T = 0.63;//0.62;
  float T_F = 100;//100;
  float A_i = 1;
  float max_f = 77.6595;

  //auto dNdpT = myFunc(100, beta_T, T_F, A_i);
  float maxeta = 2.5;

  TFile *file = TFile::Open("thermal_back_10.root","recreate");
  //cn4->Write();
  TH1F* h1 = new TH1F("h1", Form("Uniform #phi"), 20, 0., 2*M_PI);
  TH1F* h2 = new TH1F("h2", Form("Uniform #eta"), 20, -maxeta, maxeta);
  TH1F* h3 = new TH1F("h3", Form("p_T dist"), 20, 0., 1.);
  TH1F* h4 = new TH1F("h4", Form("p_T dist"), 20, 0., 1.);
  TH2F* h2D = new TH2F("h2D", Form("p_T dist"), 20, 0., 2*M_PI, 20, -maxeta, maxeta);

  std::mt19937 generator(0);
  std::uniform_real_distribution <double>  distr(0, 2*M_PI);
  std::uniform_real_distribution <double>  distr2(-maxeta, maxeta);
  std::uniform_real_distribution <double>  distr3(0, 1000.);
  std::uniform_real_distribution <double>  distr4(0, 1.);

  vector <double> all_pt;
  vector <double> dN;
  vector <fastjet::PseudoJet> listjets;
  for (int i = 0; i < 5000; i++){
    double phi = distr(generator);
    double eta = distr2(generator);
    
    h1->Fill(phi);
    h2->Fill(eta);
  
    float pt;
    bool done=false;
    while (done==false){
      double rand3 = distr3(generator);
      double rand4 = distr4(generator);
      if (rand4 > myFunc(rand3, beta_T, T_F, A_i)/max_f) {
        //cout << "draw again" << endl;
        continue;
      } else{
        //cout << " ---------- " << endl;
        pt = rand3;
        done=true;
      }
    }
    h3->Fill(pt*pow(10,-3));
    all_pt.push_back(pt*pow(10,-3));
    h4->Fill(myFunc(pt*pow(10,-3), beta_T, T_F, A_i));
    dN.push_back(myFunc(pt*pow(10,-3), beta_T, T_F, A_i));
    
    //Create 4-vector (frmula (2.32) in 1901.10342)
    float E  = pt*cosh(eta)*pow(10,-3);
    float px = pt*cos(phi)*pow(10,-3);
    float py = pt*sin(phi)*pow(10,-3);
    float pz = pt*sinh(eta)*pow(10,-3);

    fastjet::PseudoJet jet = fastjet::PseudoJet(px, py, pz, E);

    listjets.push_back(jet);

    h2D->Fill(phi, eta, pt*pow(10,-3));
  }

  //cout << listjets.size() << endl;
  for (int i; i<listjets.size(); ++i){
    h4->Fill(listjets[i].pt());
  }
  
  TCanvas *c1 = new TCanvas("c1");
  h1->SetStats(0);
  h1->GetXaxis()->SetTitle("#phi");
  h1->Draw();
  c1->Write();

  TCanvas *c2 = new TCanvas("c2");
  h2->SetStats(0);
  h2->GetXaxis()->SetTitle("#eta");
  h2->Draw();
  c2->Write();

  TCanvas *c3 = new TCanvas("c3");
  h3->SetStats(0);
  h3->GetXaxis()->SetTitle("p_T (GeV)");
  h3->Draw();
  c3->Write();

  TCanvas *c4 = new TCanvas("c4");
  h4->SetStats(0);
  h4->GetXaxis()->SetTitle("k_{T}/p_{T} ");
  h4->Draw();
  c4->Write();

  Int_t n = 5000;
  Double_t x[n], y[n];

  for (Int_t i=0;i<n;i++) {
      x[i] = all_pt[i];
      y[i] = dN[i];
   }


  TCanvas *c10 = new TCanvas("c10");
  TGraph *gr1 = new TGraph(n, x, y);
  gr1->Draw();
  c10->Write();


  cout << std::accumulate(all_pt.begin(), all_pt.end(), 0.0) / all_pt.size() << endl;
  /*
  TFile *myFile = TFile::Open("pt_spectr_2.root");
  TCanvas *cn = (TCanvas *)myFile->Get("antikt_SD_ng;1");
  TCanvas *cn1 = (TCanvas *)myFile->Get("kt_SD_ng;1");
  TCanvas *cn12 = (TCanvas *)myFile->Get("SD_ng;1");
  TGraphAsymmErrors *ae = (TGraphAsymmErrors*)cn->GetPrimitive("h11_9");
  TGraphAsymmErrors *ae1 = (TGraphAsymmErrors*)cn->GetPrimitive("h11_6");
  TGraphAsymmErrors *ae21 = (TGraphAsymmErrors*)cn->GetPrimitive("h11_3");
  TGraphAsymmErrors *ae31 = (TGraphAsymmErrors*)cn1->GetPrimitive("h11_8");
  TGraphAsymmErrors *ae41 = (TGraphAsymmErrors*)cn1->GetPrimitive("h11_5");
  TGraphAsymmErrors *ae51 = (TGraphAsymmErrors*)cn1->GetPrimitive("h11_2");
  TGraphAsymmErrors *ae61 = (TGraphAsymmErrors*)cn12->GetPrimitive("h11_7");
  TGraphAsymmErrors *ae71 = (TGraphAsymmErrors*)cn12->GetPrimitive("h11_4");
  TGraphAsymmErrors *ae81 = (TGraphAsymmErrors*)cn12->GetPrimitive("h11_1");


  TFile *file = TFile::Open("ptspect_overGeV.root","recreate");
  TCanvas *cn4 = new TCanvas("woBack","", 600, 500);
  ae51->SetLineColor(kRed);
  ae51->SetLineStyle(1);
  ae51->SetLineWidth(1);
  ae51->SetMarkerColor(kRed);
  ae51->SetMarkerStyle(kFullSquare);
  ae51->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7,  without background}{SoftDrop z_{cut}=0.1, #beta=1}"));
  ae51->Draw("P");
  ae81->SetLineColor(kBlue);
  ae81->SetLineStyle(1);
  ae81->SetLineWidth(1);
  ae81->SetMarkerColor(kBlue);
  ae81->SetMarkerStyle(kFullDiamond);
  ae81->Draw("P same");
  ae21->SetLineColor(kBlack);
  ae21->SetLineStyle(1);
  ae21->SetLineWidth(1);
  ae21->SetMarkerColor(kBlack);
  ae21->SetMarkerStyle(kFullCircle);
  ae21->Draw("P same");
  TLegend *ln1 = new TLegend(0.5, 0.8, 0.8, 0.88);
  ln1->AddEntry(ae81, "C/A");
  ln1->AddEntry(ae21, "Anti-k_{t}");
  ln1->AddEntry(ae51, "k_{t}");
  ln1->SetFillColor(0);
  ln1->SetBorderSize(0);
  ln1->Draw();
  cn4->SetLogy();
  cn4->SetLeftMargin(0.2);
  cn4->Write();
  cn4->SaveAs("aaa.png");

  TCanvas *cn5 = new TCanvas("wBack1");
  ae41->SetLineColor(kRed);
  ae41->SetLineStyle(1);
  ae41->SetLineWidth(1);
  ae41->SetMarkerColor(kRed);
  ae41->SetMarkerStyle(kFullSquare);
  ae41->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, with 750 background particles }{Blast Wave Model #beta_{T}=0.52, T_{F}=120, SoftDrop z_{cut}=0.1, #beta=1}"));
  ae41->Draw("P");
  ae71->SetLineColor(kBlue);
  ae71->SetLineStyle(1);
  ae71->SetLineWidth(1);
  ae71->SetMarkerColor(kBlue);
  ae71->SetMarkerStyle(kFullDiamond);
  ae71->Draw("P same");
  ae1->SetLineColor(kBlack);
  ae1->SetLineStyle(1);
  ae1->SetLineWidth(1);
  ae1->SetMarkerColor(kBlack);
  ae1->SetMarkerStyle(kFullCircle);
  ae1->Draw("P same");
  TLegend *ln2 = new TLegend(0.5, 0.8, 0.8, 0.88);
  ln2->AddEntry(ae71, "C/A");
  ln2->AddEntry(ae1, "Anti-k_{t}");
  ln2->AddEntry(ae41, "k_{t}");
  ln2->SetFillColor(0);
  ln2->SetBorderSize(0);
  ln2->Draw();
  cn5->SetLogy();
  cn5->SetLeftMargin(0.2);
  cn5->Write();

  TCanvas *cn6 = new TCanvas("wBack2");
  ae31->SetLineColor(kRed);
  ae31->SetLineStyle(1);
  ae31->SetLineWidth(1);
  ae31->SetMarkerColor(kRed);
  ae31->SetMarkerStyle(kFullSquare);
  ae31->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, with 5000 background particles}{Blast Wave Model #beta_{T}=0.63, T_{F}=100, SoftDrop z_{cut}=0.1, #beta=1}"));
  ae31->Draw("P");
  ae61->SetLineColor(kBlue);
  ae61->SetLineStyle(1);
  ae61->SetLineWidth(1);
  ae61->SetMarkerColor(kBlue);
  ae61->SetMarkerStyle(kFullDiamond);
  ae61->Draw("P same");
  ae->SetLineColor(kBlack);
  ae->SetLineStyle(1);
  ae->SetLineWidth(1);
  ae->SetMarkerColor(kBlack);
  ae->SetMarkerStyle(kFullCircle);
  ae->Draw("P same");
  TLegend *ln3 = new TLegend(0.5, 0.8, 0.8, 0.88);
  ln3->AddEntry(ae61, "C/A");
  ln3->AddEntry(ae, "Anti-k_{t}");
  ln3->AddEntry(ae31, "k_{t}");
  ln3->SetFillColor(0);
  ln3->SetBorderSize(0);
  ln3->Draw();
  cn6->SetLogy();
  cn6->SetLeftMargin(0.2);
  cn6->Write();
  */

  delete file;
    

  // Done.
  return 0;
}
