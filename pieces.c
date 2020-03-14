// Making chess pieces using raymarching and constructive solid geometry
// Copy + paste all of this code into shadertoy and run it

// Defines
#define STEP_MAX 200
#define DIST_MAX 100.0
#define EPSILON 0.001

const vec3 BISHOP_POS = vec3(0, 0, 20);

// Signed distance functions defined here:
float sphere_sdf( vec3 p, float r)
{
  return length(p) - r;
}

float floor_sdf(vec3 p){
    vec4 plane = vec4(0, 1, 0, 0);
	return dot(p, plane.xyz) - plane.w;
}
float rounded_cylinder_sdf( vec3 p, float ra, float rb, float h )
{
  vec2 d = vec2( length(p.xz)- 2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}
float rounded_cone_sdf( vec3 p, float r1, float r2, float h )
{
  vec2 q = vec2( length(p.xz), p.y );
    
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);
  float k = dot(q,vec2(-b,a));
    
  if( k < 0.0 ) return length(q) - r1;
  if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
  return dot(q, vec2(a,b) ) - r1;
}
float tri_prism_sdf( vec3 p, vec2 h )
{
    const float k = sqrt(3.0);
    h.x *= 0.5*k;
    p.xy /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p.xy=vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    float d1 = length(p.xy)*sign(-p.y)*h.x;
    float d2 = abs(p.z)-h.y;
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}
/**
 * Rotation matrix around the X axis.
 */
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}
/**
 * Rotation matrix around the Y axis.
 */
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

/**
 * Rotation matrix around the Z axis.
 */
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

vec3 translate(vec3 p, float x, float y, float z){
	return p + vec3(x, y, z);   
}


float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); 
}

float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }

float opSmoothIntersection( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h); 
}

float opUnion( float d1, float d2 ) {  return min(d1,d2); }


float bishop_base(vec3 p){
    float cone = rounded_cylinder_sdf(p, .55,.1,.25);
    float ring = rounded_cylinder_sdf(p - vec3(0,0.1,0), .25,.003,.08);
    return opSmoothUnion(cone,ring, 0.2);
   
}
float bishop_shaft(vec3 p){
    float ring = rounded_cylinder_sdf(p - vec3(0,2.5,0), .35,.005,.1);
    float ring2 = rounded_cylinder_sdf(p - vec3(0,2.7,0), .25,.003,.08);
    float ring3 = rounded_cylinder_sdf(p - vec3(0,2.9,0), .2,.001,.06);
    float shaft_tip = opUnion(opUnion(ring, ring2), ring3);
   
    float cone = rounded_cone_sdf(p, .4,.1, 5.);
    float ring4 = rounded_cylinder_sdf(p - vec3(0,0.4,0), .25,.003,.08);
    float shaft_base = opUnion(opSmoothUnion(cone, ring4, 0.3), ring4);
    return opSmoothUnion(shaft_base, shaft_tip,0.5);
}

float bishop_tip(vec3 p){
    float sphere = sphere_sdf(p, 0.55);
    vec3 rot = rotateZ(46.0) * (p - vec3(-0.4,0.2,0));
    float t_prism = tri_prism_sdf(rot, vec2(0.65,0.5));
    return opSmoothSubtraction(t_prism,sphere, 0.1);
    
}

float bishop_sdf(vec3 p){
 
    float tip = bishop_tip(p-vec3(BISHOP_POS.x, BISHOP_POS.y + 5.0, BISHOP_POS.z));
    
    float base = bishop_base(p-vec3(BISHOP_POS.x, BISHOP_POS.y + 0.5, BISHOP_POS.z));
   
    float shaft = bishop_shaft(p - vec3(BISHOP_POS.x, BISHOP_POS.y + 1.0, BISHOP_POS.z));
    
    float peen = opSmoothUnion(opSmoothUnion(base, shaft,.5),tip,0.5);
    return peen;
}



// Combine all the distance functions for the scene in this function
float scene_sdf(vec3 p){
    float bishop = bishop_sdf(p);
    float plane = floor_sdf(p);
    return min(plane, bishop);
    
}


// Perform ray marching by finding the min distance ray can travel without hitting anything and iterating
float ray_march(vec3 cam_pos, vec3 cam_dir){
	float t_near = 0.0;
    for(int i = 0; i < STEP_MAX; i++){
    	vec3 p = cam_pos + cam_dir * t_near; // t_near is how far we can go along ray without hitting object
        float dist = scene_sdf(p);
        t_near += dist;
        // Check if we missed entirely or hit something
        if(t_near > DIST_MAX || dist < EPSILON){
        	break;  // > DIST_MAX then we missed all objects, less than EPSILON, we hit an object   
        }
    }
    
    return t_near;
}




// Get normal by approximating the gradient at some point in the scene
vec3 normal_at(vec3 p){
    float dist = scene_sdf(p);
    return normalize(dist - vec3( 
        scene_sdf(p - vec3(0.01, 0, 0)),
        scene_sdf(p - vec3(0, 0.01, 0)),
        scene_sdf(p - vec3(0, 0 , 0.01))
    ));
}



// Add simple point lights to illuminate the scene
float get_light(vec3 p){
    float l1_intensity = 1.5;
    vec3 l1 = vec3(-3, 4, 8);
    //l1.xyz += vec3(sin(iTime), sin(iTime), cos(iTime));
    vec3 l1_dir = normalize(l1 - p);  // Direction vector from the light to the point
    vec3 norm = normal_at(p);
    
    float kd = dot(norm,l1_dir);  // Calculate diffuse color intensity as dot product of light direction and surface normal
    
    //float shadow = ray_march(p + norm * EPSILON * 2., l1_dir);  // MUST ADD Epsilon to ensure don't accidently hit the floor
    //if(shadow < length(l1 - p)){  // Hit something between light and point so we're in a shadow
      //  l1_intensity = 0.1;
    //}
    
    kd = clamp(kd, 0.0, 1.0) * l1_intensity;
    
    return kd;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = (fragCoord - (0.5) * iResolution.xy)/iResolution.y;

    vec3 cam_pos = vec3(0,1,0);
    vec3 cam_dir = vec3(uv.x, uv.y, 1);
    
    float t = ray_march(cam_pos, cam_dir);
    vec3 point = cam_pos + cam_dir * t;  // Point in the scene (for shading purposes)
    float kd = get_light(point);
    
    
    // Rendering to screen
    vec3 col = vec3(kd);
    fragColor = vec4(col,1.0);
}
