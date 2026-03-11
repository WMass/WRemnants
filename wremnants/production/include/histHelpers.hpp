#ifndef WREMNANTS_HIST_HELPERS_H
#define WREMNANTS_HIST_HELPERS_H

#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <THn.h>
#include <cstdio>
#include <cstdlib> //as stdlib.h
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "utils.hpp"

// from narf
#include "histutils.hpp"

namespace wrem {

void fillTHNplus1fromTHn(THnD &thnp1, THnD &thn, int binLow = 0,
                         int binHigh = -1) {
  int ndim = thn.GetNdimensions();

  if (binHigh < 0)
    binHigh = thnp1.GetAxis(ndim)->GetNbins() + 1; // up to overflow

  if (binLow > binHigh)
    throw std::range_error("Invalid inputs! binLow must be less than binHigh");

  std::vector<int> binLookup(ndim + 1, 0);
  for (Long64_t globalBin = 0; globalBin < thn.GetNbins(); globalBin++) {
    double binContent = thn.GetBinContent(globalBin, binLookup.data());
    double binError = thn.GetBinError(globalBin);
    for (int iNewDim = binLow; iNewDim <= binHigh; iNewDim++) {
      binLookup[ndim] = iNewDim;
      Long64_t globalBinTHnP1 = thnp1.GetBin(binLookup.data());
      thnp1.SetBinContent(globalBinTHnP1, binContent);
      thnp1.SetBinError(globalBinTHnP1, binError);
    }
  }

  // return thnp1;
}

void fillTH3fromTH3part(
    TH3 &h3out, TH3 &h3in, int xbinLow = 1, int ybinLow = 1, int zbinLow = 1,
    int xbinHigh = -1, int ybinHigh = -1, int zbinHigh = -1, int xoffset = 0,
    int yoffset = 0,
    int zoffset = 0, // shift bins by these offsets when reading input
    bool fillWithValuePlusError = false,
    double scaleError = 1.0, // fill with binContent + scaleError*binError
                             // (negative scaleError to subtract)
    bool fillWithError = false) {

  if (xbinHigh < 0)
    xbinHigh = h3out.GetNbinsX();
  if (ybinHigh < 0)
    ybinHigh = h3out.GetNbinsY();
  if (zbinHigh < 0)
    zbinHigh = h3out.GetNbinsZ();

  double content = 0.0;
  double error = 0.0;

  for (Int_t iz = zbinLow; iz <= zbinHigh; iz++) {
    for (Int_t iy = ybinLow; iy <= ybinHigh; iy++) {
      for (Int_t ix = xbinLow; ix <= xbinHigh; ix++) {
        double err = scaleError *
                     h3in.GetBinError(ix + xoffset, iy + yoffset, iz + zoffset);
        if (fillWithValuePlusError) {
          content =
              h3in.GetBinContent(ix + xoffset, iy + yoffset, iz + zoffset) +
              err;
          error = std::abs(err);
        } else if (fillWithError) {
          content = err;
          error = 0.0;
        } else {
          content =
              h3in.GetBinContent(ix + xoffset, iy + yoffset, iz + zoffset);
          error = std::abs(err);
        }
        h3out.SetBinContent(ix, iy, iz, content);
        h3out.SetBinError(ix, iy, iz, error);
      }
    }
  }
}

void fillTH3fromTH3(TH3 &th3out, TH3 &th3in, int bin3outStart = 1,
                    int bin3inLow = 1, int bin3inHigh = -1,
                    bool copyError = true) {

  if (bin3inHigh < 0)
    bin3inHigh = th3in.GetNbinsZ(); // no overflow

  if (bin3inLow > bin3inHigh)
    throw std::range_error("Invalid inputs! binLow must be less than binHigh");

  int bin3outEnd = bin3outStart + (bin3inHigh - bin3inLow);
  if (bin3outEnd > th3out.GetNbinsZ())
    bin3outEnd = th3out.GetNbinsZ();

  int th3inBinZ = bin3inLow;
  double content = 0.0;
  double error = 0.0;

  for (Int_t iz = bin3outStart; iz <= bin3outEnd; iz++) {
    for (Int_t ix = 1; ix <= th3out.GetNbinsX(); ix++) {
      for (Int_t iy = 1; iy <= th3out.GetNbinsY(); iy++) {
        content = th3in.GetBinContent(ix, iy, th3inBinZ);
        th3out.SetBinContent(ix, iy, iz, content);
        if (copyError)
          th3out.SetBinError(ix, iy, iz, th3in.GetBinError(ix, iy, th3inBinZ));
      }
    }
    th3inBinZ++;
  }
}

// should write another function to broadcast into THn
void broadCastTH2intoTH3(TH3 &th3, TH2 &th2, int binLow = 0, int binHigh = -1,
                         bool zeroError = false) {

  if (binHigh < 0)
    binHigh = 1 + th3.GetNbinsZ(); // up to overflow
  double binContent = 0.0;
  double binError = 0.0;
  for (Int_t ix = 1; ix <= th3.GetNbinsX(); ix++) {
    for (Int_t iy = 1; iy <= th3.GetNbinsY(); iy++) {
      binContent = th2.GetBinContent(ix, iy);
      binError = zeroError ? 0.0 : th2.GetBinError(ix, iy);
      for (Int_t iz = binLow; iz <= binHigh; iz++) {
        th3.SetBinContent(ix, iy, iz, binContent);
        th3.SetBinError(ix, iy, iz, binError);
      }
    }
  }
}

template <typename T>
bool cropNegatives(T &h, Long64_t nbins, double cropValue, bool cropError) {
  bool hasCroppedBins = false;
  for (Long64_t globalBin = 0; globalBin < nbins; globalBin++) {
    double binContent = h.GetBinContent(globalBin);
    if (binContent < 0.0) {
      hasCroppedBins = true;
      h.SetBinContent(globalBin, cropValue);
      if (cropError)
        h.SetBinError(globalBin, cropValue);
    }
  }
  return hasCroppedBins;
}

bool cropNegativeContent(THnD &h, bool silent = false, bool cropError = false,
                         double cropValue = 0.0001) {

  Long64_t nbins = h.GetNbins();
  bool hasCroppedBins = cropNegatives<THnD>(h, nbins, cropValue, cropError);
  if (not silent and hasCroppedBins)
    std::cout << "Cropping negative bins for histogram " << h.GetName()
              << std::endl;

  return hasCroppedBins;
}

bool cropNegativeContent(TH1 &h, bool silent = false, bool cropError = false,
                         double cropValue = 0.0001) {

  Long64_t nbins = h.GetNcells();
  double integral = h.Integral();
  double integralNonNeg = 0.0;

  bool hasCroppedBins = cropNegatives<TH1>(h, nbins, cropValue, cropError);
  if (not silent and hasCroppedBins) {
    integralNonNeg = h.Integral();
    std::cout << h.GetName() << ": original integral = " << integral
              << " changed by " << integralNonNeg / integral << " (new/old)"
              << std::endl;
  }
  return hasCroppedBins;
}

template <typename T>
void initializeHistogram(T &h, Long64_t nbins, double init = 0.0) {
  for (Long64_t globalBin = 0; globalBin < nbins; globalBin++) {
    h.SetBinContent(globalBin, init);
    h.SetBinError(globalBin, 0.0);
  }
}
void initializeRootHistogram(TH1 &h, double init = 0.0) {
  Long64_t nbins = h.GetNcells();
  initializeHistogram<TH1>(h, nbins, init);
}

template <typename T>
void setHistogramError(T &h, Long64_t nbins, double init = 0.0) {
  for (Long64_t globalBin = 0; globalBin < nbins; globalBin++) {
    h.SetBinError(globalBin, 0.0);
  }
}
void setRootHistogramError(TH1 &h, double init = 0.0) {
  Long64_t nbins = h.GetNcells();
  setHistogramError<TH1>(h, nbins, init);
}

void setRootHistFromHistError(TH1 &h, TH1 &hin, bool poisson = false) {
  Long64_t nbins = h.GetNcells();
  for (Long64_t globalBin = 0; globalBin < nbins; globalBin++) {
    h.SetBinContent(globalBin, poisson ? std::sqrt(hin.GetBinContent(globalBin))
                                       : hin.GetBinError(globalBin));
    h.SetBinError(globalBin, 0.0);
  }
}

void makeHistStatUncertaintyRatio(TH1 &h, TH1 &hin) {
  Long64_t nbins = h.GetNcells();
  for (Long64_t globalBin = 0; globalBin <= nbins; globalBin++) {
    h.SetBinContent(globalBin, hin.GetBinError(globalBin) /
                                   std::sqrt(hin.GetBinContent(globalBin)));
    h.SetBinError(globalBin, 0.0);
  }
}

TH2D projectTH2FromTH3(TH3 &hist3D, const char *name, size_t binStart,
                       size_t binEnd = 0) {
  if (binEnd == 0)
    binEnd = binStart;
  hist3D.GetZaxis()->SetRange(binStart, binEnd);
  // Order yx matters to have consistent axes!
  TH2D hist2D = *static_cast<TH2D *>(hist3D.Project3D("yxe"));
  hist2D.SetName(name);
  return hist2D;
}

template <typename T> std::array<T, 2> envelopHists(std::vector<T> &hists) {

  std::string name = hists.at(0).GetName();
  T hup = *static_cast<T *>(hists.at(0).Clone(name + "Up"));
  T hdown = *static_cast<T *>(hup.Clone(name + "Down"));

  for (size_t i = 0; i < hup.GetNcells(); i++) {

    double minc = hup.GetBinContent(i);
    double mine = hup.GetBinError(i);
    double maxc = hup.GetBinContent(i);
    double maxe = hup.GetBinError(i);

    for (auto &h : hists) {
      double cont = h.GetBinContent(i);
      if (cont > maxc) {
        maxc = cont;
        maxe = h.GetBinError(i);
      } else if (cont < minc) {
        minc = cont;
        mine = h.GetBinError(i);
      }
    }

    hup.SetBinContent(i, maxc);
    hup.SetBinError(i, maxe);
    hdown.SetBinContent(i, minc);
    hdown.SetBinError(i, mine);
  }

  return {hup, hdown};
}

} // namespace wrem

#endif
