<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.8">
  <compound kind="file">
    <name>AnalogDiscoveryIO.cc</name>
    <path></path>
    <filename>AnalogDiscoveryIO_8cc.html</filename>
    <includes id="AnalogDiscoveryIO_8hh" name="AnalogDiscoveryIO.hh" local="yes" import="no" module="no" objc="no">AnalogDiscoveryIO.hh</includes>
    <namespace>gramsballoon</namespace>
  </compound>
  <compound kind="file">
    <name>AnalogDiscoveryIO.hh</name>
    <path></path>
    <filename>AnalogDiscoveryIO_8hh.html</filename>
    <class kind="class">gramsballoon::AnalogDiscoveryIO</class>
    <namespace>gramsballoon</namespace>
    <member kind="define">
      <type>#define</type>
      <name>ANALOG_OUT_STOP</name>
      <anchorfile>AnalogDiscoveryIO_8hh.html</anchorfile>
      <anchor>a612ec37cb2ceb243d5549b5e2a1fdae8</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ANALOG_OUT_START</name>
      <anchorfile>AnalogDiscoveryIO_8hh.html</anchorfile>
      <anchor>a51a93fe1ca2fefc6bf6e681dd5ae5a06</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ANALOG_OUT_APPLY</name>
      <anchorfile>AnalogDiscoveryIO_8hh.html</anchorfile>
      <anchor>a2c8e46c79bf6b49d204082beba45f57a</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>main.cc</name>
    <path></path>
    <filename>main_8cc.html</filename>
    <includes id="AnalogDiscoveryIO_8hh" name="AnalogDiscoveryIO.hh" local="yes" import="no" module="no" objc="no">AnalogDiscoveryIO.hh</includes>
    <member kind="function">
      <type>bool</type>
      <name>check_ManualTrigger</name>
      <anchorfile>main_8cc.html</anchorfile>
      <anchor>a198d19194ab4fdaad4b410d3b20291e4</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>check_PPS_extTrigger</name>
      <anchorfile>main_8cc.html</anchorfile>
      <anchor>a157803c7d1b615bbebe82e438443bc42</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>main</name>
      <anchorfile>main_8cc.html</anchorfile>
      <anchor>abf9e6b7e6f15df4b525a2e7705ba3089</anchor>
      <arglist>(int argc, char const *argv[])</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>AnalogDiscoveryIO</name>
    <filename>classAnalogDiscoveryIO.html</filename>
  </compound>
  <compound kind="class">
    <name>gramsballoon::AnalogDiscoveryIO</name>
    <filename>classgramsballoon_1_1AnalogDiscoveryIO.html</filename>
    <member kind="function">
      <type></type>
      <name>AnalogDiscoveryIO</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>ae96b12964441fefebeefbf17eec3e674</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>initialize</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>a18705095c9be6ec9e650e3056efd5e54</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>connect</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>af4c66d4ab7dea9ab57f096617fbbada6</anchor>
      <arglist>(int device_id)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>connect</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>a095036872980d918e75ba278618c1b84</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>setupAnalogOut</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>aebc981021e91411c2409a0b88d13b278</anchor>
      <arglist>(int device_id, int channel, double init_value=0.0, std::string signalType=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>setupAnalogIn</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>aff669ec194729e289fec835d06d4abc9</anchor>
      <arglist>(int device_id, int channel, double freq, int buf_size, double range, double offset)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>setVoltage</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>ac5d62e7c812930cab9664383c479b902</anchor>
      <arglist>(int device_id, int channel, double voltage, int sleep)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>finalize</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>a63e81880c3d8d77525fc8dab10536095</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>NumDevices</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>a926c8cebf18667f5862ba3733150ab36</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; HDWF &gt; &amp;</type>
      <name>HandlerList</name>
      <anchorfile>classgramsballoon_1_1AnalogDiscoveryIO.html</anchorfile>
      <anchor>ae50f6bfdfd5ac91fcc02eadc8ef08931</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>gramsballoon</name>
    <filename>namespacegramsballoon.html</filename>
    <class kind="class">gramsballoon::AnalogDiscoveryIO</class>
    <member kind="function">
      <type>std::string</type>
      <name>convert_vector_string</name>
      <anchorfile>namespacegramsballoon.html</anchorfile>
      <anchor>a6b829afce8a4b2abae819c14db400a2d</anchor>
      <arglist>(const std::vector&lt; char &gt; &amp;vec)</arglist>
    </member>
  </compound>
</tagfile>
