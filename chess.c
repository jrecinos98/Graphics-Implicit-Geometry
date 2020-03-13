// Making chess pieces using raymarching and constructive solid geometry
// Copy + paste all of this code into shadertoy and run it

// Defines
#define STEP_MAX 200
#define DIST_MAX 100.0
#define EPSILON 0.001


// Signed distance functions defined here:
float sphere_sdf( vec3 p, vec3 c, float r)
{
  return length(p - c) - r;
}

float cyclinder_sdf(vec3 p, 

float floor_sdf(vec3 p){
    vec4 plane = vec4(0, 1, 0, 0);
	return dot(p, plane.xyz) - plane.w;
}

vec3 translate(vec3 p, float x, float y, float z){
	return p + vec3(x, y, z);   
}

// Combine all the distance functions for the scene in this function
float scene_sdf(vec3 p){
    float sphere = sphere_sdf(p, vec3(0, 1, 8), 1.);
    float plane = floor_sdf(p);
    return min(plane, sphere);
    
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
    float l1_intensity = 1.;
    vec3 l1 = vec3(0, 6, 9);
    vec3 l1_dir = normalize(l1 - p);  // Direction vector from the light to the point
    vec3 norm = normal_at(p);
    
    float kd = dot(norm, l1_dir);  // Calculate diffuse color intensity as dot product of light direction and surface normal
    
    float shadow = ray_march(p + norm * EPSILON * 2., l1_dir);  // MUST ADD Epsilon to ensure don't accidently hit the floor
    if(shadow < length(l1 - p)){  // Hit something between light and point so we're in a shadow
        l1_intensity = 0.1;
    }
    
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