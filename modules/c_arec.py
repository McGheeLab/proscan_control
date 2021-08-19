
import sys
import os
import winsound
import time
from datetime import datetime
import math
from matplotlib import pyplot as plt

from modules.m1_basic_control import *
from modules.m99_sim_serial import spo






#######################################################
class c_arec:
  #
  def __init__(self):
    print("Remember, use q to quit any time.")
    self.name = []  # area name
    self.px = []    # x pos
    self.py = []    # y pos
    self.notes = [] 
    self.name_prefix = 'a'
    #
    self.n_area = 0
    #
    print("DANGER:  Backup system not implemented yet.")
    print("  If you set areas and then load, you will loose your data.")
  #
  def prefix(self,p=None):
    if p != None:
      self.name_prefix = p
    print("Using new prefix'"+self.name_prefix+"'")
    print("  Next area name will be: ",self.next_name())
  #
  def clear_areas(self):
    self.name = []  # area name
    self.px = []    # x pos
    self.py = []    # y pos
    self.notes = [] 
    #
    self.n_area = 0
  #
  def ls(self):  # list
    for i in range(self.n_area):
      line = str(i)+'. '
      line += "("+str(self.px[i])+','+str(self.py[i])+")"
      line += "  ["+self.name[i]+"]"
      line += "  - " + self.notes[i]
      print(line)
    print("n_area: ", self.n_area)
  #
  def next_name(self):
    auto_n = len(self.name_prefix)+3
    next_i = 0   # The first possible will actually be 1
    n0 = len(self.name_prefix)
    for i in range(self.n_area):
      if len(self.name[i]) != auto_n:  continue
      if not self.name[i].startswith( self.name_prefix ):  continue
      # print("::> ", self.name[i])
      oor = 0  #  index out of range
      for j in range(3):  # 3 digits
        acode = ord(self.name[i][n0+j])
        if acode < 48:  oor = 1    # ascii 0
        if acode > 57:  oor = 1    # ascii 9
      if oor == 1:  continue
      cunum = int(self.name[i][n0:])
      if cunum > next_i:  next_i = cunum
    next_i += 1
    return self.name_prefix+'{0:03d}'.format( next_i )
  #
  def set(self, name=None):
    #
    if name == None:  uname = self.next_name()
    else:             uname = name
    i = self.n_area
    self.n_area += 1
    #
    self.name.append( uname )
    self.px.append( 0 )
    self.py.append( 0 )
    self.notes.append( "" )
    print("Setting ", self.name[i])
    #
    ###
    cbuf() # Make sure the buffer is clear.
    send = bytes( "p\r\n".encode() )
    spo.write( send )
    serda = spo.readline()
    ll = serda.decode("Ascii").split(',')
    self.px[i] = int(ll[0])
    self.py[i] = int(ll[1])
  #
  def pos(self, mode=None):
    # report the current position
    cbuf() # Make sure the buffer is clear.
    send = bytes( "p\r\n".encode() )
    spo.write( send )
    serda = spo.readline()
    ll = serda.decode("Ascii").split(',')
    posx = int(ll[0])
    posy = int(ll[1])
    posz = int(ll[2])
    if mode == 'print':  print("x y z: ",posx, posy, posz)
    return posx, posy, posz
  #
  def go(self,name):
    j = -1
    for i in range(self.n_area):
      if name == self.name[i]:
        j = i
    if j == -1:
      print("Couldn't find area.")
      return
    ###
    x = self.px[j]
    y = self.py[j]
    ouline = "g"
    ouline += " {0:d}".format( x )
    ouline += " {0:d}".format( y )
    ouline += "\r\n"
    send = bytes( ouline.encode() )
    spo.write( send )
  #
  def load(self):
    print("Loading data...")
    #
    fname_base =  "arec.data"
    fname_default = "config/"+fname_base
    fname_user = "user/"+fname_base
    if os.path.isfile( fname_user ):
      fname = fname_user
      print("Found user file.")
    else:
      fname = fname_default
      print("Using default file.")
    #
    self.clear_areas()
    #
    print("  Loading: ", fname )
    f = open(fname)
    for l in f:
      l = l.strip()
      if len(l) == 0:  continue
      if l[0] == '#':  continue
      ###
      ll = l.split(';')
      self.px.append( int(ll[1].strip()) )
      self.py.append( int(ll[2].strip()) )
      self.name.append( ll[3].strip() )
      if len(ll) > 4:
        self.notes.append( ll[4].strip() )
      else:
        self.notes.append("")
      ###
    f.close()
    self.n_area = len(self.name)
    print("  Done.")
  #
  def save(self, ufname=None):
    if self.n_area == 0:
      print("Nothing to save.")
      return
    fname_base =  "arec.data"
    if ufname != None:
      fname_base = ufname
    fname_user = "user/"+fname_base
    fname = fname_user
    print("Saving "+fname+" ...")
    #
    fz = open(fname, 'w')
    for i in range(self.n_area):
      ou = str(i)
      ou += " ; " + str(self.px[i])
      ou += " ; " + str(self.py[i])
      ou += " ; " + self.name[i]
      if len(self.notes[i])>0:
        ou += " ; " + self.notes[i]
      ou += '\n'
      fz.write(ou)
    fz.close()
  #
  def backup(self):
    ts_dto = datetime.now()
    ts = ts_dto.strftime("%Y%m%d_%H%M%S")
    ufname = "arec_"+ts+".data"
    self.save( ufname )
  #
  def ls_rel(self):
    # Show the positions of all the areas relative
    # to the current stage position.
    cbuf() # Make sure the buffer is clear.
    send = bytes( "p\r\n".encode() )
    spo.write( send )
    serda = spo.readline()
    ll = serda.decode("Ascii").split(',')
    cupx = int(ll[0])
    cupy = int(ll[1])
    #
    # relative positions.
    repx = []
    repy = []
    reran = []  # relative range (distance)
    reazi = []  # relative azimuth
    #
    for i in range(self.n_area):
      repx.append(0)
      repy.append(0)
      reran.append(0)
      reazi.append(0)
      #
      repx[i] = -( self.px[i] - cupx ) # neg because inverted stage x axis
      repy[i] = self.py[i] - cupy
      reran[i] = math.hypot( repx[i], repy[i] )
      reazi[i] = math.atan2( repy[i], repx[i] )
      reazi[i] *= 180 / math.pi
      reazi[i] = 90 - reazi[i]
      if reazi[i] < 0:  reazi[i] += 360
    #
    for i in range(self.n_area):
      line = str(i)+'. '
      line += "pos_xy("+str(self.px[i])+','+str(self.py[i])+")"
      line += "  "
      # line += "range_azim("+str(reran[i])+','+str(reazi[i])+")"
      line += "azim_range({0:0.0f}".format(reazi[i])
      # line += "deg,"
      # line += "�," # fails
      line += u'\u00B0'  # degree symbol
      line += ","
      line += "{0:0.0f}".format(reran[i])
      # line += ",um)"
      # line += ","
      line += u'\u00B5'  # greek mu
      line += "m)"
      line += "  ["+self.name[i]+"]"
      # line += "  - " + self.notes[i]
      uline = line.encode('utf-8')
      print(line)
    print("azimuths use geo reference frame and degrees.")
    print("current pos: ", cupx, cupy)
    print("n_area: ", self.n_area)
    #
  #
  def plot(self, plot_save=0, plot_grc=0):
    # plot_save:  0 no, 1 yes
    # plot_grc:  0 no, 1 yes plot current stage position
    if self.n_area == 0:
      print("No areas defined.")
      return
    # The edge will be either a circle "e+c"/n_s
    # or a rectangle "e+r"/n_rec.
    n_s = 0
    n_rec = 0
    # Center will be at (cx, cy)
    cx = 0
    cy = 0
    for i in range(self.n_area):
      if self.name[i].startswith("e+c"):
        n_s += 1
        cx += self.px[i]
        cy += self.py[i]
      if self.name[i].startswith("e+r"):
        n_rec += 1
    if n_s != 4 and n_rec != 4:
      print("Didn't find four e+?### names for N W S E.")
      print("  n_s   (e+c): ", n_s)
      print("  n_rec (e+r): ", n_rec)
      return
    #
    if n_s == 4:
      cx /= 4
      cy /= 4
      circ_r = 0
      for i in range(self.n_area):
        if self.name[i].startswith("e+c"):
          circ_r += math.hypot( cx - self.px[i], cy - self.py[i] )
      #
      circ_r /= 4
      circ_n_seg = 80
      circ_n_pnt = circ_n_seg+1
      circ_x = []
      circ_y = []
      circ_dang = math.pi * 2 / circ_n_seg
      for i in range(circ_n_pnt):
        ang = circ_dang * i
        circ_x.append( cx + circ_r * math.cos( ang ) )
        circ_y.append( cy + circ_r * math.sin( ang ) )
    #
    if n_rec == 4:
      rec_xmin = None
      rec_xmax = None
      rec_ymin = None
      rec_ymax = None
      for i in range(self.n_area):
        if self.name[i].startswith("e+r"):
          if rec_xmin == None:  rec_xmin = self.px[i]
          if rec_ymin == None:  rec_ymin = self.py[i]
          if rec_xmax == None:  rec_xmax = self.px[i]
          if rec_ymax == None:  rec_ymax = self.py[i]
          if self.px[i] < rec_xmin:  rec_xmin = self.px[i]
          if self.py[i] < rec_ymin:  rec_ymin = self.py[i]
          if self.px[i] > rec_xmax:  rec_xmax = self.px[i]
          if self.py[i] > rec_ymax:  rec_ymax = self.py[i]
      # grr:  graph of rectangle edge.
      grrx = [rec_xmin, rec_xmax, rec_xmax, rec_xmin, rec_xmin]
      grry = [rec_ymin, rec_ymin, rec_ymax, rec_ymax, rec_ymin]
    #
    # gra will be the areas.
    grax = []
    gray = []
    graa = []  # annotation
    for i in range(self.n_area):
      if not self.name[i].startswith("e+"):
        grax.append( self.px[i] )
        gray.append( self.py[i] )
        graa.append( self.name[i] )
    gra_n = len(grax)
    #
    # grc:  graph current position.
    grcx = []
    grcy = []
    if plot_grc == 1:
      cbuf() # Make sure the buffer is clear.
      send = bytes( "p\r\n".encode() )
      spo.write( send )
      serda = spo.readline()
      ll = serda.decode("Ascii").split(',')
      cupx = int(ll[0])
      cupy = int(ll[1])
      #
      grcx.append(cupx)
      grcy.append(cupy)
    #
    if n_s == 4:
      plt.plot( circ_x, circ_y,
        color='#000099'
        )
    if n_rec == 4:
      plt.plot( grrx, grry,
        color='#000099'
        )
    if plot_grc == 1:
      plt.plot( grcx, grcy,
        marker='+',
        markerfacecolor='None',
        markeredgecolor='#aa0000',
        linestyle='None'
        )
    plt.plot( grax, gray,
      marker='o',
      markerfacecolor='None',
      markeredgecolor='#ff0000',
      linestyle='None'
      )
    #
    ax = plt.gca()
    gra_style = dict(size=8, color='black')
    for i in range(gra_n):
      ha='left'
      ax.text( grax[i], gray[i], ' '+graa[i], ha=ha, **gra_style )
    #
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='box')
    #
    ts_dto = datetime.now()
    ts = ts_dto.strftime("%Y%m%d_%H%M%S")
    ufname = "user/arec_"+ts+".png"
    #
    if plot_save == 1:  plt.savefig( ufname )
    plt.show()
  #
  def run(self):   # human user system
    ####################################
    while( 1 ):
      #########################
      print()
      print("Entering arec user run.")
      print("  Use q to quit.")
      while True:
        print()
        uline = input("u>> ")
        if uline == 'q':
          print()
          return
        elif uline == 'ls':  self.ls()
        elif uline == 'ls rel':
          self.ls_rel()
        elif uline == 'load':  self.load()
        elif uline == 'save':  self.save()
        elif uline == 'backup':  self.backup()
        elif uline == 'plot ?':
          print("Usage:")
          print("  plot          # Just show the plot.")
          print("  plot cp       # Also plot current position.")
          print("  plot save     # Also save a PNG of the graph.")
          print("  plot save cp  # Save PNG, include current pos.")
          print("  plot cp save  # Save PNG, include current pos.")
        elif uline == 'plot':  self.plot()
        elif uline == 'plot save':  self.plot( plot_save=1)
        elif uline == 'plot cp':  # cp:  also plot current position
          self.plot(plot_grc=1)
        elif uline == 'plot save cp':  self.plot( plot_save=1, plot_grc=1)
        elif uline == 'plot cp save':  self.plot( plot_save=1, plot_grc=1)
        elif uline == 'clear':  self.clear_areas()
        elif uline == 'pos':  self.pos(mode='print')
        elif uline.startswith( 'set' ):
          if uline == 'set':
            self.set()
          elif len(uline) > len('set '):
            aname = uline[4:]
            self.set(aname)
          else:
            print("Problem using set.  Unexpected uline.")
            print("  uline: ", uline)
        elif uline.startswith('go '):
          if len(uline) <= len('go '):
            print("No entered entered.")
          else:
            # go abc
            # 01234567
            aname = uline[3:]
            self.go( aname )
        elif uline.startswith('prefix'):
          if uline == 'prefix':
            self.prefix()  # Just show the current prefix.
          elif len(uline) <= len('prefix '):  # note the extra space
            print("No prefix entered.")
            self.prefix()
          else:
            # prefix abc
            # 01234567
            pre = uline[7:]
            self.prefix( pre )
        else:
          print("Unrecognized input.")
      #########################
    ####################################
  #
#######################################################


arec = c_arec()



