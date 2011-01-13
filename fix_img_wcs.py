import numpy as np
import pywcs
import pyfits
from numpy import sin, cos, radians

DEBUG = False

def rotate(degs):
    """Return a rotation matrix for counterclockwise rotation by ``deg`` degrees."""
    rads = radians(degs)
    s = sin(rads)
    c = cos(rads)
    return np.array([[c, -s],
                     [s,  c]])

def read_fits(name, hdu):
    """Read FITS file ``name`` with an image in specified ``hdu`` number.
    Return the image HDU, list of all HDUs and the WCS object associated
    with the image HDU.
    """
    hdulist = pyfits.open(name)
    img_hdu = hdulist[hdu]
    wcs = pywcs.WCS(img_hdu.header)
    return img_hdu, hdulist, wcs

def write_fits(hdulist, name, clobber=True, checksum=True):
    """Write the ``hdulist`` to a FITS file with name ``name``."""
    hdulist.writeto(name, clobber=clobber, checksum=checksum)

def update_header_wcs(hdu, wcs):
    """Update the WCS CRVAL and CD values in the header for the given ``hdu``
    using the supplied ``wcs`` WCS object.  This assumes that the CD values
    are being used instead of the PC values (as is the case for an HST
    Multidrizzle output). 
    """
    hdr = hdu.header
    hdr['CRVAL1'] = wcs.wcs.crval[0]
    hdr['CRVAL2'] = wcs.wcs.crval[1]
    if hasattr(wcs.wcs, 'cd'):
        hdr['CD1_1'] = wcs.wcs.cd[0,0]
        hdr['CD1_2'] = wcs.wcs.cd[0,1]
        hdr['CD2_1'] = wcs.wcs.cd[1,0]
        hdr['CD2_2'] = wcs.wcs.cd[1,1]
    if hasattr(wcs.wcs, 'pc'):
        hdr['PC1_1'] = wcs.wcs.pc[0,0]
        hdr['PC1_2'] = wcs.wcs.pc[0,1]
        hdr['PC2_1'] = wcs.wcs.pc[1,0]
        hdr['PC2_2'] = wcs.wcs.pc[1,1]

class WcsModel(object):
    def __init__(self, wcs, sky, pix0):
        self.wcs = wcs   # Image WCS transformation object
        self.sky = sky   # Reference (correct) source positions in RA, Dec
        self.pix0 = pix0.flatten()  # Source pixel positions
        # Copy the original WCS CRVAL and CD values
        self.crval = wcs.wcs.crval.copy()
        if hasattr(wcs.wcs, 'cd'):
            self.cd = wcs.wcs.cd.copy()
        else:
            self.cd = wcs.wcs.pc.copy()

    def calc_pix(self, pars, x=None):
        """For the given d_ra, d_dec, and d_theta pars, update the WCS
        transformation and calculate the new pixel coordinates for each
        reference source position.

        The "x=None" parameter is because Sherpa passes an extra "X"
        argument, which in this case we always ignore.
        """
        d_ra, d_dec, d_theta = pars
        self.wcs.wcs.crval = self.crval + np.array([d_ra, d_dec]) / 3600.0
        if hasattr(self.wcs.wcs, 'cd'):
            self.wcs.wcs.cd = np.dot(rotate(d_theta), self.cd)
        else:
            self.wcs.wcs.pc = np.dot(rotate(d_theta), self.cd)
        pix = self.wcs.wcs_sky2pix(self.sky, 1)
        if DEBUG:
            print 'pix =', pix.flatten()
            print 'pix0 =', self.pix0.flatten()
        return pix.flatten()

    def calc_resid2(self, pars):
        """Return the squared sum of the residual difference between the
        original pixel coordinates and the new pixel coords (given offset
        specified in ``pars``)

        This gets called by the scipy.optimize.fmin function.
        """
        pix = self.calc_pix(pars)
        resid2 = np.sum((self.pix0 - pix)**2)  # assumes uniform errors
        if DEBUG:
            print 'resid2 =', resid2
        return resid2

def match_wcs(wcs_img, sky_img, sky_ref, opt_alg='scipy'):
    """Adjust ``wcs_img`` (CRVAL{1,2} and CD{1,2}_{1,2}) using a rotation and linear
    offset so that ``coords_img`` matches ``coords_ref``.

    :param sky_img: list of (world_x, world_y) [aka RA, Dec] coords in input image
    :param sky_ref: list of reference (world_x, world_y) coords to match
    :param wcs_img: pywcs WCS object for input image

    :returns: d_ra, d_dec, d_theta
    """
    pix_img = wcs_img.wcs_sky2pix(sky_img, 1)
    wcsmodel = WcsModel(wcs_img, sky_ref, pix_img)
    y = np.array(pix_img).flatten()
    
    if opt_alg == 'sherpa':
        x = np.arange(len(y))
        import sherpa.astro.ui as ui
        ui.load_user_model(wcsmodel.calc_pix, 'wcsmod')
        ui.add_user_pars('wcsmod', ['d_ra', 'd_dec', 'd_theta'])
        wcsmod.d_ra = 0.0
        wcsmod.d_dec = 0.0
        wcsmod.d_theta = 0.0
        ui.load_arrays(1, x, y, np.ones(len(y)))
        ui.set_model(1, wcsmod)
        ui.set_method('simplex')
        ui.fit()
    else:
        import scipy.optimize
        x0 = np.array([0.0, 0.0, 0.0])
        d_ra, d_dec, d_theta = scipy.optimize.fmin(wcsmodel.calc_resid2, x0)
        print 'Scipy fit values:', d_ra, d_dec, d_theta

    return wcsmodel.wcs
    
def fix_img_wcs(infile, outfile, sky_ref, sky_img, opt_alg='scipy', hdu=1):
    """
    Adjust the WCS transform in FITS file ``infile`` so that the sources
    positions given in ``sky_img`` most closely match the "correct" values in
    ``sky_ref``.  The FITS image is assumed to be in the given ``hdu`` number
    (default=1).  The updated image (along with any other HDUs) are written out
    to ``outfile``.  The optimization algorithm can be "scipy"
    (scipy.optimize.fmin) or "sherpa".
    """
    img_hdu, hdulist, wcs_img = read_fits(infile, hdu)
    new_wcs = match_wcs(wcs_img, sky_img, sky_ref, opt_alg)
    update_header_wcs(img_hdu, new_wcs)
    write_fits(hdulist, outfile)

def test():
    # List of (RA, Dec) for the "reference" (correct) positions for 4 sources
    sky_ref = [(130.0048, 29.8197),
               (130.00679, 29.81488),
               (130.01521, 29.81453),
               (130.01099, 29.81773),
               ]
    # List of (RA, Dec) measured in the HST image for the same 4 sources
    sky_img = [(130.00499, 29.81962),
               (130.00693, 29.81473),
               (130.01542, 29.81432),
               (130.01117, 29.81759),
               ]

    fix_img_wcs('test.fits', 'test_fix_scipy.fits',
                sky_ref, sky_img, opt_alg='scipy', hdu=0)
    fix_img_wcs('test.fits', 'test_fix_sherpa.fits',
                sky_ref, sky_img, opt_alg='sherpa', hdu=0)
