// main72.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: basic usage; fastjet; slowjet; jet finding;

// This is a simple test program.
// It compares SlowJet, FJcore and FastJet, showing that they
// find the same jets.

#include "Pythia8/Pythia.h"
//k
// The FastJet3.h header enables automatic initialisation of
// fastjet::PseudoJet objects from Pythia8 Particle and Vec4 objects,
// as well as advanced features such as access to (a copy of)
// the original Pythia 8 Particle directly from the PseudoJet,
// and fastjet selectors that make use of the Particle properties.
// See the extensive comments in the header file for further details
// and examples.

#include "Pythia8Plugins/FastJet3.h"
//Importing ROOT
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH2F.h"

#include <time.h>
#include <iostream>
#include <vector>
#include <random>
#include <stdlib.h>
#include <chrono>
#include "fastjet/ClusterSequenceArea.hh"
#include "main72.h"

using namespace Pythia8;


int main() {
  // Number of events, generated and listed ones (for jets).
  int nEvent    = 10;

  // Select common parameters for SlowJet and FastJet analyses.
  int    power   = -1;     // -1 = anti-kT; 0 = C/A; 1 = kT.
  double R       = 0.7;    // Jet size.
  double pTMin   = 5.0;    // Min jet pT.
  double etaMax  = 2.0;    // Pseudorapidity range of detector.
  int    select  = 2;      // Which particles are included?
  int    massSet = 2;      // Which mass are they assumed to have?

  //My parameters
  vector <double> alpha_l={1., 0.1, 0.01}; //Rigidity parameter
  double alpha;
  vector <int> power_l = {1, 0}; //Type of algorithm
  int power2;
  float z_cut = 0.1; //SoftDrop symmetry cut
  int beta = 1; //SoftDrop beta

  //Histogram parameters
  double ktlargest = 0.5;
  double mlargest = 0.7;
  double nlargest = 100;
 
  // Generator. Shorthand for event.
  Pythia pythia;
  Event& event = pythia.event;

  //Pythia settings
  ostringstream pythiaset;
  pythiaset << "pythia_settings.cmd";
  pythia.readFile(pythiaset.str());
  pythia.init();

  //Set up root file and histograms
  TFile *file = TFile::Open("Qjet_wbackground_200_sub_2.root","recreate");

  TH2F* h2D = new TH2F("h2D", Form("Lund plane Fastjet"), 50, 0., 10., 50, -12., 10.);
  TH2F* h2D2 = new TH2F("h2D2", Form("Lund plane Q-jet"), 50, 0., 10, 50, -12., 12.);

  std::vector <vector <vector <vector <TH1F*> > > >  list_all = MakeListHist(25, mlargest, ktlargest, nlargest);
  std::vector <vector <vector <TH1F*> > > list_true_all = MakeListHistTrue(100, mlargest, ktlargest, nlargest);



  // Set up FastJet jet definition
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm , R, power);
  double ghost_area=0.01;
  fastjet::AreaDefinition areaDef(fastjet::active_area_explicit_ghosts, fastjet::GhostedAreaSpec(2.5, 1, ghost_area));

  //Create various list I use in my Q-jet algorithm 
  std::vector <fastjet::PseudoJet> fjInputs;
  std::vector <vector <MyClass> > finaljets;
  std::vector <vector <vector <MyClass> > > alphajets;
  std::vector <vector <vector <vector <MyClass> > > > algorjets;
  std::vector <vector <vector <vector <vector <MyClass> > > > > partjets;

  //Start a random number generator
  std::srand(0);


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

      //Save jets
      fjInputs.push_back( particleTemp);
    }
    
    // Without background
    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequence clustSeq(fjInputs, jetDef);
    inclusiveJets = clustSeq.inclusive_jets(pTMin);
    sortedJets    = sorted_by_pt(inclusiveJets);

    //Make sure a jet is created otherwise kill whole loop
    if (sortedJets.size()==0) continue;
  
    //Create particle list without background and print info
    vector <fastjet::PseudoJet> particles = sortedJets[0].constituents();
    
    cout << std::setprecision(10);
    cout << "Jet pt = " << sortedJets[0].pt() << endl;
    cout << "Jet Mass = " << sortedJets[0].m() << endl;
    cout << "Number of particles in jet: " << particles.size() << endl;


    //Get background particles, create particle list and print info
    vector <fastjet::PseudoJet> bgparticles = getbackground(750, 0.52, 120, 102.5);
    cout << "Total Number of background particles added:  " << bgparticles.size() << endl;
    vector <fastjet::PseudoJet> allparticles = GetParticleList(fjInputs, bgparticles, jetDef, areaDef,pTMin, R);

    //Get background particles, create particle list and print info
    vector <fastjet::PseudoJet> bgparticles2 = getbackground(5000, 0.63, 100, 77.6595);
    cout << "Total Number of background particles added:  " << bgparticles2.size() << endl;
    vector <fastjet::PseudoJet> allparticles2 = GetParticleList(fjInputs, bgparticles2, jetDef, areaDef,pTMin, R);

    //
   

    //Create list of lists of particles
    vector <vector <fastjet::PseudoJet> > list_particles = {particles, allparticles, allparticles2};



    //Start Q-jet process
    for (int parts=0; parts<list_particles.size(); ++parts){
      //For the different particle lists
      for (int Algorithm=0; Algorithm<2; Algorithm++){
        //For the different algorithms power={1,0}
        power2 = power_l[Algorithm];
        for (int AL=0; AL<3; AL++){
          //For the different alpha values alpha={1, 0.1, 0.01}
          auto start = std::chrono::high_resolution_clock::now();
          alpha = alpha_l[AL];
          int id, d1, d2, nrp;

          for (int N=0; N < 10000; N++){
            //For the number of Q-jets N_Qjet
            vector <int> org_index;
            vector <MyClass> myobj;
            int identity = 1;

            for (int i=0; i<list_particles[parts].size(); ++i){
              id = identity;
              org_index.push_back(id);
              ++identity;
              d1 = 0;
              d2 = 0;
              nrp = 1;
              myobj.push_back(MyClass(id, d1, d2, list_particles[parts][i], nrp));
            }
             
            vector <fastjet::PseudoJet> new_particles = list_particles[parts];

            for (int n = 1; n < list_particles[parts].size(); ++n){
              //Repeat the process until all instances has been used onec, i.e. when there is only one particle left in new_particles

              //Get the index of the particle we are clustering
              auto [i,j] = GetIndexForClustering(new_particles, R, power2, alpha);

              //Create the clustered jet
              fastjet::PseudoJet new_jet = new_particles[i];
              new_jet += new_particles[j];
              new_particles[i] = new_jet;

              //Save the clustered jet as an object
              id = identity;
              d1 = org_index[i];
              d2 = org_index[j];
              nrp = myobj[d1-1].nrparticles + myobj[d2-1].nrparticles;
              myobj.push_back(MyClass(id, d1, d2, new_particles[i], nrp));

              //Change the original index to the id of the new clustered object
              org_index[i] = id;

              //To remove the jets we clustered together (i becomes the new object, need to remove j)
              vector <int> new_i;
              vector <fastjet::PseudoJet> nn_particles;
              for (int l=0; l<new_particles.size(); ++l){
                if (l==j) continue;
                nn_particles.push_back(new_particles[l]);
                new_i.push_back(org_index[l]);
              }

              //Replece the list of org_index and new_particles with the new list without j
              new_particles.clear();
              new_particles=nn_particles;
              org_index.clear();
              org_index=new_i;
              
              //Update the identity count
              ++identity;
            }
            //Save the list of objects belonging to this Q-jet
            finaljets.push_back(myobj);
          }
          //Save the list of objects belonging to this alpha 
          alphajets.push_back(finaljets);
          finaljets.clear();
          
          //Print time it takes
          auto stop = std::chrono::high_resolution_clock::now();
          auto duration = duration_cast<std::chrono::microseconds>(stop - start);
          cout << "Time = " << duration.count()*pow(10,-6) << " s" << endl;
        }
        //Save the list of objects belonging to this algorithm
        algorjets.push_back(alphajets);
        alphajets.clear();
      }
      //Save the list of objects belonging to this list of particles
      partjets.push_back(algorjets);
      algorjets.clear();
    }
    //End of Q-jet algorithm
    cout << endl;
    

    //Save the Q-jet observables to histograms
    int id2, ds1, ds2, nrp1, nrp2;
    vector <fastjet::PseudoJet> listjet1;
    for (int part=0; part<partjets.size(); part++){
      //For the list of particles
      for (int algo=0; algo<partjets[0].size(); algo++){
        //For the algorithm
        for (int alp=0; alp<partjets[0][0].size(); alp++){
          //For the alpha
          for (int N=0; N<partjets[0][0][0].size(); N++){
            //For the number of Q-jets

            //Extract information from the object
            id2 = partjets[part][algo][alp][N][partjets[part][0][0][N].size()-1].id;
            ds1 = partjets[part][algo][alp][N][id2-1].d1;
            ds2 = partjets[part][algo][alp][N][id2-1].d2;
            
            //Save the two jets at the hardest splitting
            listjet1.push_back(partjets[part][algo][alp][N][ds1-1].jet);
            listjet1.push_back(partjets[part][algo][alp][N][ds2-1].jet);

            //Calculate observables
            float theta = listjet1[0].delta_R(listjet1[1]);
            auto [mh, m0, z, kt] = get_m(listjet1);
            
            //Fill histograms with observables
            list_all[part][algo][alp][0]->Fill(theta/R);
            list_all[part][algo][alp][1]->Fill(sqrt(mh)/partjets[part][algo][alp][N][id2-1].jet.pt());
            list_all[part][algo][alp][2]->Fill(sqrt(m0));
            list_all[part][algo][alp][3]->Fill(z);
            list_all[part][algo][alp][4]->Fill(sqrt(kt/partjets[part][algo][alp][N][id2-1].jet.pt()));

            //Apply SoftDrop
            int sdjet = SoftDrop(partjets[part][algo][alp][N], z_cut, beta, R);
            int sdd1 = partjets[part][algo][alp][N][sdjet-1].d1;
            int sdd2 = partjets[part][algo][alp][N][sdjet-1].d2;

            //Calculate SoftDrop observables
            float sdtheta = partjets[part][algo][alp][N][sdd1-1].jet.delta_R(partjets[part][algo][alp][N][sdd2-1].jet);
            auto [sdmh, sdm0, sdz, sdkt] = get_m({partjets[part][algo][alp][N][sdd1-1].jet, partjets[part][algo][alp][N][sdd2-1].jet});

            //Fill histograms with SoftDrop observables
            list_all[part][algo][alp][5]->Fill(sdtheta/R);
            list_all[part][algo][alp][6]->Fill(sdz);
            list_all[part][algo][alp][7]->Fill(sqrt(sdmh)/partjets[part][algo][alp][N][sdjet-1].jet.pt());
            list_all[part][algo][alp][8]->Fill(sqrt(sdkt)/partjets[part][algo][alp][N][sdjet-1].jet.pt());

            //Extraxt the number of particles in each of the two subjets and fill histogram with them 
            nrp1 = partjets[part][algo][alp][N][sdd1-1].nrparticles;
            nrp2 = partjets[part][algo][alp][N][sdd2-1].nrparticles;
            list_all[part][algo][alp][9]->Fill((nrp2-nrp1)/partjets[part][algo][alp][N][sdjet-1].nrparticles);

            LundPlaneQ(partjets[part][algo][alp][N], h2D2, 1, R);

            listjet1.clear();
          }
        }
      }
    }
    // Done preparing all histograms for Q-jets

    //Prepare histograms for fastjet algorithms
    for (int part=0; part<list_particles.size(); ++part){
      //For the list of particles
      for (int pow=0; pow<power_l.size(); ++pow){
        //For the algorithm 
        jetDef.set_extra_param(power_l[pow]);

        //Cluster jets
        vector <fastjet::PseudoJet> inclusiveJets2;
        fastjet::ClusterSequence clustSeq2(list_particles[part], jetDef);
        inclusiveJets2 = clustSeq2.inclusive_jets(pTMin);
        fastjet::PseudoJet jet = inclusiveJets2[0];

        //Save two subjets of the hardest splitting
        std::vector <fastjet::PseudoJet > listjet	=jet.exclusive_subjets(2);

        //Calculate observables of the hardest splitting 
        float theta2 = listjet[0].delta_R(listjet[1]);
        auto [mh2, m02, z2, kt2] = get_m(listjet);

        //Apply SoftDrop
        auto[sdjet,nrSD] = SoftDrop2({jet}, z_cut, beta, R);
        std::vector <fastjet::PseudoJet > sdlistjet	=sdjet[0].exclusive_subjets(2);

        //Calculate SoftDrop variables
        float sdtheta2 = sdlistjet[0].delta_R(sdlistjet[1]);
        auto [sdmh2, sdm02, sdz2, sdkt2] = get_m(sdlistjet);


        //Extract number of particles in each of the two subjets
        int nrp3 = sdlistjet[0].constituents().size();
        int nrp4 = sdlistjet[1].constituents().size();

        //Fill all histograms
        for (int i=0; i<partjets[part][0][0].size(); ++i){
          list_true_all[part][pow][0]->Fill(theta2/R);
          list_true_all[part][pow][1]->Fill(sqrt(mh2)/jet.pt());
          list_true_all[part][pow][2]->Fill(sqrt(m02));
          list_true_all[part][pow][3]->Fill(z2);
          list_true_all[part][pow][4]->Fill(sqrt(kt2)/jet.pt()); 
          list_true_all[part][pow][5]->Fill(sdtheta2/R);
          list_true_all[part][pow][6]->Fill(sdz2);
          list_true_all[part][pow][7]->Fill(sqrt(sdmh2)/sdjet[0].pt());
          list_true_all[part][pow][8]->Fill(sqrt(sdkt2)/sdjet[0].pt());
          list_true_all[part][pow][9]->Fill((nrp4-nrp3)/sdjet[0].constituents().size());
        }

        //Lund plane
        LundPlane(jet, h2D, 1, R);

        listjet.clear(); 
      }
    }
    //Done preparing all fastjet histograms
    
    //Make sure to only do one jet at a time by breaking here
    break;

  // End of event loop.
  }

  // Draw histograms
  
  //Prepare title strings, etc.
  vector <string> labels = {"#theta", "m_{h}", "m_{0}", "z", "k_{t}", "softdrop #theta", "softdrop z", "softdrop m_{h}", "softdrop m_{0}", "N2-N1"};
  vector <string> types = {"k_{t}", "C/A"};
  vector <string> wbg = {"without background", "with background 1", "with background 2"};
  vector <EColor> color = {kBlack, kRed, kBlue};

  //Do the plotting

  //Plot alphas in the same plot, algorithm and particle list separated (only for two particle lists)
  //Plot1(list_true_all, list_all, color, labels, types,  wbg, alpha_l);

  //Plot the algorithms in same plot, particle list and alphas separated
  //Plot2(list_true_all, list_all, color, labels, types,  wbg, alpha_l);

  //Plot particle lists in same plot, algorithm and alphas seperated
  //Want to see how change with more background
  Plot3(list_true_all, list_all, color, labels, types,  wbg, alpha_l);


 

  
  TCanvas *c20 = new TCanvas("c20");
  h2D->SetStats(0);
  h2D->GetYaxis()->SetTitle("ln kt/pt");
  h2D->GetXaxis()->SetTitle("ln R/#theta");
  h2D->Draw("COL");
  c20->Write();
  
  TCanvas *c21 = new TCanvas("c21");
  h2D2->SetStats(0);
  h2D2->GetYaxis()->SetTitle("ln kt/pt");
  h2D2->GetXaxis()->SetTitle("ln R/#theta");
  h2D2->Draw("COL");
  c21->Write();
  
  


  
  /*
  //Anti-kt
  cout << "Anti-kt" << endl;
  cout << Form("Alpha = %.2f", alpha_l[0]) << endl;
  float mean = h6->GetMean();
  float Lambda =  h6->GetRMS();
  cout << Form("Volatility = %.5f", Lambda/mean) << endl;
  cout << Form("Alpha = %.2f", alpha_l[1]) << endl;
  float mean2 = h26->GetMean();
  float Lambda2 =  h26->GetRMS();
  cout << Form("Volatility = %.5f", Lambda2/mean2) << endl;
  cout << Form("Alpha = %.2f", alpha_l[2]) << endl;
  float mean3 = h36->GetMean();
  float Lambda3 =  h36->GetRMS();
  cout << Form("Volatility = %.5f", Lambda3/mean3) << endl;
  cout << endl;
  cout << "C/A" << endl;
  cout << Form("Alpha = %.2f", alpha_l[0]) << endl;
  float mean4 = h46->GetMean();
  float Lambda4 =  h46->GetRMS();
  cout << Form("Volatility = %.5f", Lambda4/mean4) << endl;
  cout << Form("Alpha = %.2f", alpha_l[1]) << endl;
  float mean5 = h56->GetMean();
  float Lambda5 =  h56->GetRMS();
  cout << Form("Volatility = %.5f", Lambda5/mean5) << endl;
  cout << Form("Alpha = %.2f", alpha_l[2]) << endl;
  float mean6 = h66->GetMean();
  float Lambda6 =  h66->GetRMS();
  cout << Form("Volatility = %.5f", Lambda6/mean6) << endl;
  
  */
  delete file;
  // Done.

  return 0;
}
